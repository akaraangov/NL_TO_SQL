import streamlit as st
from sqlalchemy import create_engine, text
import config
import requests
import json
import torch
import os
import re
import sqlparse

# -------------------------------
# DB SCHEMA FUNCTIONS (PostgreSQL)
# -------------------------------
def get_schema_info(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """))
        schema_info = ""
        for row in result:
            schema_info += f"Table: {row.table_name}\n - {row.column_name} ({row.data_type})\n"
    return schema_info

def extract_schema_dict(engine):
    schema = {}
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """))
        for row in result:
            table = row.table_name
            column = row.column_name
            if table not in schema:
                schema[table] = []
            schema[table].append(column)
    return schema

# -------------------------------
# PROMPT CONSTRUCTION
# -------------------------------
def construct_prompt(schema_info, user_question, errors=None, schema_dict=None):
    base = f"""
You are an expert SQL generator.

Here is the database schema:
{schema_info}

User request: "{user_question}"

IMPORTANT:
- Use ONLY the table and column names listed above.
- DO NOT invent table or column names.
- If the requested information does not exist, respond clearly:
"No such table or column exists in the current database schema for this query."
- Output ONLY the SQL query or the specific message without explanations or formatting.
"""
    if errors and schema_dict:
        allowed = "\n".join([f"{table}: {', '.join(cols)}" for table, cols in schema_dict.items()])
        base += f"\n⚠️ Validation errors from previous attempt: {errors}\nAllowed tables and columns:\n{allowed}\n"
    return base

# -------------------------------
# SPLIT THINKING AND SQL
# -------------------------------
def split_thinking_and_sql(response_text):
    if "</think>" in response_text:
        parts = response_text.split("</think>")
        thinking = parts[0].replace("<think>", "").strip()
        sql_or_msg = parts[1].strip()
    else:
        thinking = ""
        sql_or_msg = response_text.strip()
    return thinking, sql_or_msg

# -------------------------------
# LLM SQL GENERATION
# -------------------------------
def generate_sql_ollama(prompt, model_name):
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 2048}}
    response = requests.post(config.OLLAMA_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['response']
    return f"Error from Ollama: {response.text}"

# -------------------------------
# SQL PARSING & VALIDATION (FIXED)
# -------------------------------
def extract_tables_columns(sql):
    tables = set(re.findall(r'\bFROM\s+([a-zA-Z_][\w]*)', sql, re.IGNORECASE) +
                 re.findall(r'\bJOIN\s+([a-zA-Z_][\w]*)', sql, re.IGNORECASE))
    # Extract columns more flexibly (handles aliases, prefixes, and SELECT *)
    select_match = re.search(r'\bSELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    columns = set()
    if select_match:
        col_str = select_match.group(1)
        col_candidates = re.split(r',\s*', col_str)
        for c in col_candidates:
            col_name = c.strip().split()[-1] if " as " in c.lower() else c.strip().split(".")[-1]
            if col_name != "*":
                col_name = re.sub(r'["`\']', '', col_name)
                columns.add(col_name)
    return tables, columns

def validate_sql(sql, schema):
    if "No such table or column exists" in sql:
        return False, "LLM determined the requested table/column does not exist."

    tables, columns = extract_tables_columns(sql)
    invalid_tables = tables - set(schema.keys())
    all_columns = {col for cols in schema.values() for col in cols}
    invalid_columns = {col for col in columns if col not in all_columns}

    errors = []
    if invalid_tables:
        errors.append(f"Invalid tables: {', '.join(invalid_tables)}")
    if invalid_columns:
        errors.append(f"Invalid columns: {', '.join(invalid_columns)}")
    return len(errors) == 0, "\n".join(errors)

# -------------------------------
# SQL EXECUTION
# -------------------------------
def execute_sql(sql, engine):
    with engine.connect() as conn:
        try:
            result = conn.execute(text(sql))
            keys = result.keys()
            return [dict(zip(keys, row)) for row in result.fetchall()]
        except Exception as e:
            return {"error": str(e)}

# -------------------------------
# MODEL DETECTION
# -------------------------------
def detect_local_models(models_dir):
    return [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))] if os.path.exists(models_dir) else []

def get_ollama_models():
    import subprocess
    result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, text=True)
    return [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]

# -------------------------------
# STREAMLIT APP
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
engine = create_engine(config.DB_URI)
schema_info = get_schema_info(engine)
schema_dict = extract_schema_dict(engine)

st.title("💬 NL-to-SQL with PostgreSQL Validation")
model_mode = st.radio("Model source:", ["Ollama", "Local"])
model_list = get_ollama_models() if model_mode == "Ollama" else detect_local_models(config.LLM_MODELS_DIR)
selected_model = st.selectbox("Choose model:", model_list) if model_list else None
user_input = st.text_input("Enter your query:")
max_attempts = st.number_input("Max retry attempts", min_value=1, max_value=5, value=2, step=1)
if st.checkbox("Show DB Schema"):
    st.text(schema_info)

if st.button("Run Query") and user_input and selected_model:
    if not schema_dict:
        st.warning("⚠️ No tables found in the database. Please check your connection.")
    else:
        attempt, response_text, sql_or_msg, errors = 1, None, None, None
        while attempt <= max_attempts:
            st.write(f"🔍 Attempt {attempt}...")
            prompt = construct_prompt(schema_info, user_input, errors, schema_dict)

            if model_mode == "Ollama":
                response_text = generate_sql_ollama(prompt, selected_model)
            else:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                @st.cache_resource
                def load_model(path):
                    t, m = AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
                    return t, m
                tokenizer, model = load_model(selected_model)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(DEVICE)
                outputs = model.generate(inputs['input_ids'], max_new_tokens=1024, do_sample=False)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            thinking, sql_or_msg = split_thinking_and_sql(response_text)

            if thinking:
                st.info(f"🧠 Model Reasoning:\n{thinking}")
            st.code(sql_or_msg, language="sql")

            valid, validation_msg = validate_sql(sql_or_msg, schema_dict)
            st.text(validation_msg)

            if "No such table or column exists" in sql_or_msg:
                st.warning(sql_or_msg)
                break
            elif valid:
                st.write("⚙️ Executing SQL...")
                result = execute_sql(sql_or_msg, engine)
                if isinstance(result, dict) and "error" in result:
                    st.error(f"Execution error: {result['error']}")
                else:
                    st.dataframe(result)
                break
            else:
                errors = validation_msg
                if attempt == max_attempts:
                    st.error("⛔️ Query failed after validation retries.")
                attempt += 1
