import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable
import numpy as np
import pandas as pd
import sqlite3
import json
from fpdf import FPDF
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import hashlib
import hmac
import base64
from sklearn.ensemble import IsolationForest
import yfinance as yf
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_community.llms import OpenAI

# ======================
# Configuration
# ======================
st.set_page_config(
    page_title="OptiSpend PRO+ - AI Financial Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ayushgoel135/AIvest',
        'Report a bug': "https://github.com/ayushgoel135/AIvest/issues",
        'About': "# OptiSpend PRO+ - AI-Powered Financial Optimization Suite"
    }
)

# Custom CSS with dark mode support
st.markdown("""
<style>
    :root {
        --primary: #2563EB;
        --secondary: #059669;
        --tax: #DC2626;
        --investment: #7C3AED;
        --insurance: #D97706;
        --warning: #F59E0B;
        --danger: #DC2626;
        --dark-bg: #1E293B;
        --dark-card: #334155;
        --dark-text: #F8FAFC;
    }

    /* Light mode styles */
    .main {
        background-color: #f8fafc;
        color: #1E293B;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2563EB 0%, #1E40AF 100%);
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }

    .highlight {
        background-color: var(--secondary);
        color: white;
        padding: 0.2em 0.4em;
        border-radius: 6px;
        font-weight: 500;
    }

.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
}

.leaderboard-table th {
    background-color: #2563EB;
    color: white;
    padding: 0.75rem;
    text-align: left;
}

.leaderboard-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #E5E7EB;
}

.leaderboard-table tr:nth-child(even) {
    background-color: #F3F4F6;
}

.leaderboard-table tr:hover {
    background-color: #E5E7EB;
}

.medal-gold {
    color: gold;
    font-weight: bold;
}

.medal-silver {
    color: silver;
    font-weight: bold;
}

.medal-bronze {
    color: #cd7f32; /* bronze */
    font-weight: bold;
}
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5em;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        color: #1E293B;
    }

    .tax-card {
        border-left: 4px solid var(--tax);
    }

    .investment-card {
        border-left: 4px solid var(--investment);
    }

    .insurance-card {
        border-left: 4px solid var(--insurance);
    }

    .health-card {
        border-left: 4px solid var(--secondary);
    }

    .warning-card {
        border-left: 4px solid var(--warning);
    }

    .stProgress > div > div > div {
        background-color: var(--primary) !important;
    }

    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
    }

    .stTextInput > div > div > input {
        border-radius: 8px !important;
    }

    .metric-positive {
        color: #059669;
    }

    .metric-negative {
        color: #DC2626;
    }

    .recommendation-item {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #F3F4F6;
    }

    .tax-breakdown-container {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
    }

    .tax-slab {
        flex: 1;
        padding: 0.5rem;
        margin: 0 0.2rem;
        border-radius: 5px;
        text-align: center;
        color: black;
        font-weight: bold;
    }

    .health-spider {
        width: 100%;
        height: 300px;
        margin: 1rem 0;
    }

    .business-feature {
        border-left: 4px solid #2563EB;
        padding-left: 1rem;
        margin: 1rem 0;
    }

    .report-header {
        background-color: #2563EB;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }

    .transaction-card {
        border-left: 4px solid #4F46E5;
        margin-bottom: 0.5rem;
    }

    .fraud-alert {
        border-left: 4px solid #DC2626;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }

    .goal-progress {
        height: 20px;
        border-radius: 10px;
        background: #E5E7EB;
        margin: 0.5rem 0;
    }

    .goal-progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #2563EB 0%, #059669 100%);
    }

    /* Dark mode styles */
    [data-theme="dark"] .main {
        background-color: var(--dark-bg);
        color: black;
    }

    [data-theme="dark"] .card {
        background-color: var(--dark-card);
        color: black;  }

    .recommendation-item {
        background-color: white;
        color: black;    
        }

    [data-theme="dark"] .st-bw {
        color: black !important;
    }

    [data-theme="dark"] .stTextInput > div > div > input,
    [data-theme="dark"] .stNumberInput > div > div > input,
    [data-theme="dark"] .stSelectbox > div > div > select {
        background-color: #334155;
        color: black;
    }

    [data-theme="dark"] .stDataFrame {
        background-color: #334155;
        color: black;
    }

    [data-theme="dark"] .stTable {
        color: black;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        color: black;
        background-color: #E5E7EB;
        border-radius: 10px 10px 0 0;
        border: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #D1D5DB;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: black;
    }

    [data-theme="dark"] .stTabs [data-baseweb="tab"] {
        color: black;
        background-color: #334155;
    }

    [data-theme="dark"] .stTabs [data-baseweb="tab"]:hover {
        background-color: #475569;
    }

    [data-theme="dark"] .stTabs [aria-selected="true"] {
        background-color: #1D4ED8;
        color: black;
    }
</style>
""", unsafe_allow_html=True)


# ======================
# Database Setup (Enhanced)
# ======================
def init_db():
    conn = sqlite3.connect('optispend_pro_plus.db')
    c = conn.cursor()

    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id TEXT PRIMARY KEY, 
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  business_type TEXT,
                  turnover REAL,
                  email TEXT,
                  password_hash TEXT,
                  salt TEXT,
                  two_fa_enabled BOOLEAN DEFAULT 0,
                  dark_mode BOOLEAN DEFAULT 0)''')

    c.execute('''CREATE TABLE IF NOT EXISTS financial_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  income REAL,
                  expenses REAL,
                  savings REAL,
                  investments REAL,
                  liabilities REAL,
                  insurance_cover REAL,
                  insurance_premium REAL,
                  tax_regime TEXT,
                  age INTEGER,
                  dependents INTEGER,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS leaderboard
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  score REAL,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS spending_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  category TEXT,
                  amount REAL,
                  happiness INTEGER,
                  essential BOOLEAN,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS optimizations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  category TEXT,
                  current_amount REAL,
                  optimized_amount REAL,
                  happiness INTEGER,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS financial_health
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  total_score REAL,
                  savings_score REAL,
                  investment_score REAL,
                  debt_score REAL,
                  insurance_score REAL,
                  expense_score REAL,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS business_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  tool_name TEXT,
                  input_data TEXT,
                  result_data TEXT,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    # New tables for enhanced features
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  description TEXT,
                  amount REAL,
                  category TEXT,
                  source TEXT,
                  is_fraud BOOLEAN DEFAULT 0,
                  month INTEGER,
                  year INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS investment_portfolio
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  asset_type TEXT,
                  asset_name TEXT,
                  quantity REAL,
                  buy_price REAL,
                  current_value REAL,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS financial_goals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  goal_name TEXT,
                  target_amount REAL,
                  current_amount REAL,
                  target_date DATE,
                  priority INTEGER,
                  achieved BOOLEAN DEFAULT 0,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS debts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  debt_name TEXT,
                  principal REAL,
                  interest_rate REAL,
                  monthly_payment REAL,
                  remaining_term INTEGER,
                  lender TEXT,
                  is_active BOOLEAN DEFAULT 1,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS team_members
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  email TEXT,
                  role TEXT,
                  access_level TEXT,
                  invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  accepted BOOLEAN DEFAULT 0,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS ai_recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  recommendation_type TEXT,
                  recommendation_text TEXT,
                  priority INTEGER,
                  implemented BOOLEAN DEFAULT 0,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    conn.commit()
    return conn


# ======================
# Security Functions
# ======================
def generate_salt():
    return base64.b64encode(os.urandom(16)).decode('utf-8')


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()


def verify_password(stored_hash, password, salt):
    return hmac.compare_digest(stored_hash, hash_password(password, salt))


def get_user_id():
    """Get the current user's ID from session state"""
    if 'user_id' in st.session_state:
        return st.session_state.user_id
    return None


def save_financial_data(user_id, data):
    """Save current financial data to database with month/year"""
    conn = init_db()
    c = conn.cursor()

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Check if data exists for this month/year
    c.execute('''SELECT id FROM financial_data 
                 WHERE user_id = ? AND month = ? AND year = ?''',
              (user_id, current_month, current_year))
    exists = c.fetchone()

    if exists:
        # Update existing record
        c.execute('''UPDATE financial_data 
                     SET income = ?, expenses = ?, savings = ?, investments = ?,
                         liabilities = ?, insurance_cover = ?, insurance_premium = ?,
                         tax_regime = ?, age = ?, dependents = ?
                     WHERE id = ?''',
                  (data['income'], data['expenses'], data['savings'], data['investments'],
                   data['liabilities'], data['insurance_cover'], data['insurance_premium'],
                   data['tax_regime'], data['age'], data['dependents'], exists[0]))
    else:
        # Insert new record
        c.execute('''INSERT INTO financial_data 
                     (user_id, income, expenses, savings, investments, liabilities, 
                      insurance_cover, insurance_premium, tax_regime, age, dependents, month, year)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id,
                   data['income'],
                   data['expenses'],
                   data['savings'],
                   data['investments'],
                   data['liabilities'],
                   data['insurance_cover'],
                   data['insurance_premium'],
                   data['tax_regime'],
                   data['age'],
                   data['dependents'],
                   current_month,
                   current_year))

    conn.commit()


def save_spending_data(user_id, spending_data):
    """Save current spending data to database with month/year"""
    conn = init_db()
    c = conn.cursor()

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Delete existing records for this month/year
    c.execute('''DELETE FROM spending_data 
                 WHERE user_id = ? AND month = ? AND year = ?''',
              (user_id, current_month, current_year))

    for category, data in spending_data.items():
        c.execute('''INSERT INTO spending_data 
                     (user_id, category, amount, happiness, essential, month, year)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, category, data['amount'], data['happiness'], data.get('essential', False),
                   current_month, current_year))

    conn.commit()


def save_optimization_results(user_id, current_data, optimized_data):
    """Save optimization results to database with month/year"""
    conn = init_db()
    c = conn.cursor()

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Delete existing records for this month/year
    c.execute('''DELETE FROM optimizations 
                 WHERE user_id = ? AND month = ? AND year = ?''',
              (user_id, current_month, current_year))

    for category in current_data:
        c.execute('''INSERT INTO optimizations 
                     (user_id, category, current_amount, optimized_amount, happiness, month, year)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (user_id,
                   category,
                   current_data[category]['amount'],
                   optimized_data[category],
                   current_data[category]['happiness'],
                   current_month,
                   current_year))

    conn.commit()


def update_leaderboard(user_id, score):
    """Update leaderboard with current month's score"""
    conn = init_db()
    c = conn.cursor()

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Check if already exists
    c.execute('''SELECT id FROM leaderboard 
                 WHERE user_id = ? AND month = ? AND year = ?''',
              (user_id, current_month, current_year))
    exists = c.fetchone()

    if exists:
        c.execute('''UPDATE leaderboard SET score = ? 
                     WHERE id = ?''', (score, exists[0]))
    else:
        c.execute('''INSERT INTO leaderboard 
                     (user_id, score, month, year)
                     VALUES (?, ?, ?, ?)''',
                  (user_id, score, current_month, current_year))
    conn.commit()


def get_leaderboard(limit=10):
    """Get top scores with user info"""
    conn = init_db()
    query = '''
        SELECT u.email, l.score, l.month, l.year 
        FROM leaderboard l
        JOIN users u ON l.user_id = u.user_id
        ORDER BY l.score DESC
        LIMIT ?
    '''
    return pd.read_sql(query, conn, params=(limit,))


def get_user_rank(user_id):
    """Get user's current rank"""
    conn = init_db()
    c = conn.cursor()

    # Get current month/year
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Get all scores for current period
    c.execute('''
        SELECT l.user_id, l.score
        FROM leaderboard l
        WHERE l.month = ? AND l.year = ?
        ORDER BY l.score DESC
    ''', (current_month, current_year))

    scores = c.fetchall()

    # Find user's rank
    for rank, (uid, score) in enumerate(scores, 1):
        if uid == user_id:
            return rank, len(scores), score

    return None, len(scores), None


def save_health_score(user_id, health_data):
    """Save financial health score to database with month/year"""
    conn = init_db()
    c = conn.cursor()

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Check if data exists for this month/year
    c.execute('''SELECT id FROM financial_health 
                 WHERE user_id = ? AND month = ? AND year = ?''',
              (user_id, current_month, current_year))
    exists = c.fetchone()

    if exists:
        # Update existing record
        c.execute('''UPDATE financial_health 
                     SET total_score = ?, savings_score = ?, investment_score = ?,
                         debt_score = ?, insurance_score = ?, expense_score = ?
                     WHERE id = ?''',
                  (health_data['total_score'],
                   health_data['components']['savings']['score'],
                   health_data['components']['investments']['score'],
                   health_data['components']['debt']['score'],
                   health_data['components']['insurance']['score'],
                   health_data['components']['expenses']['score'],
                   exists[0]))
    else:
        # Insert new record
        c.execute('''INSERT INTO financial_health 
                     (user_id, total_score, savings_score, investment_score, 
                      debt_score, insurance_score, expense_score, month, year)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id,
                   health_data['total_score'],
                   health_data['components']['savings']['score'],
                   health_data['components']['investments']['score'],
                   health_data['components']['debt']['score'],
                   health_data['components']['insurance']['score'],
                   health_data['components']['expenses']['score'],
                   current_month,
                   current_year))

    conn.commit()


def save_business_tool_data(user_id, tool_name, input_data, result_data):
    """Save business tool usage data"""
    conn = init_db()
    c = conn.cursor()

    current_month = datetime.now().month
    current_year = datetime.now().year

    c.execute('''INSERT INTO business_data 
                 (user_id, tool_name, input_data, result_data, month, year)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (user_id, tool_name, json.dumps(input_data), json.dumps(result_data),
               current_month, current_year))

    conn.commit()


def get_historical_data(user_id, months=12):
    """Get historical data for the user"""
    conn = init_db()

    # Get current date and calculate date range
    current_date = datetime.now()
    start_date = (current_date - relativedelta(months=months)).strftime('%Y-%m-%d')

    # Get financial data history
    financial_df = pd.read_sql('''
        SELECT * FROM financial_data 
        WHERE user_id = ? 
        AND timestamp > ?
        ORDER BY year, month
    ''', conn, params=(user_id, start_date))

    # Get spending data history
    spending_df = pd.read_sql('''
        SELECT month, year, category, amount, happiness, essential 
        FROM spending_data 
        WHERE user_id = ? 
        AND timestamp > ?
        ORDER BY year, month
    ''', conn, params=(user_id, start_date))

    # Get optimization history
    optimizations_df = pd.read_sql('''
        SELECT month, year, category, current_amount, optimized_amount, happiness
        FROM optimizations
        WHERE user_id = ?
        AND timestamp > ?
        ORDER BY year, month
    ''', conn, params=(user_id, start_date))

    # Get health score history
    health_df = pd.read_sql('''
        SELECT month, year, total_score, savings_score, investment_score,
               debt_score, insurance_score, expense_score
        FROM financial_health
        WHERE user_id = ?
        AND timestamp > ?
        ORDER BY year, month
    ''', conn, params=(user_id, start_date))

    # Get business tool usage history
    business_df = pd.read_sql('''
        SELECT month, year, tool_name, input_data, result_data
        FROM business_data
        WHERE user_id = ?
        AND timestamp > ?
        ORDER BY year, month
    ''', conn, params=(user_id, start_date))

    return {
        'financial': financial_df,
        'spending': spending_df,
        'optimizations': optimizations_df,
        'health': health_df,
        'business': business_df
    }


def get_current_month_data(user_id):
    """Get data for current month"""
    conn = init_db()
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Get financial data
    financial_data = pd.read_sql('''
        SELECT * FROM financial_data 
        WHERE user_id = ? AND month = ? AND year = ?
    ''', conn, params=(user_id, current_month, current_year))

    # Get spending data
    spending_data = pd.read_sql('''
        SELECT category, amount, happiness, essential 
        FROM spending_data 
        WHERE user_id = ? AND month = ? AND year = ?
    ''', conn, params=(user_id, current_month, current_year))

    # Convert spending data to dictionary format
    spending_dict = {}
    if not spending_data.empty:
        for _, row in spending_data.iterrows():
            spending_dict[row['category']] = {
                'amount': row['amount'],
                'happiness': row['happiness'],
                'essential': row['essential']
            }

    return {
        'financial': financial_data.iloc[0].to_dict() if not financial_data.empty else None,
        'spending': spending_dict
    }


def get_user_profile(user_id):
    """Get user profile data"""
    conn = init_db()
    profile = pd.read_sql('''
        SELECT business_type, turnover FROM users WHERE user_id = ?
    ''', conn, params=(user_id,))

    return profile.iloc[0].to_dict() if not profile.empty else None


def update_user_profile(user_id, business_type, turnover):
    """Update user profile data"""
    conn = init_db()
    c = conn.cursor()
    c.execute('''UPDATE users SET business_type = ?, turnover = ? WHERE user_id = ?''',
              (business_type, turnover, user_id))
    conn.commit()


# ======================
# Financial Calculation Engine
# ======================
class FinancialCalculator:
    @staticmethod
    def calculate_income_tax(income, regime='new', deductions=0):
        """Calculate Indian income tax liability with detailed breakdown"""
        taxable_income = max(income - deductions, 0)

        if regime == 'new':
            slabs = [
                (250000, 0),
                (500000, 0.05),
                (750000, 0.10),
                (1000000, 0.15),
                (1250000, 0.20),
                (1500000, 0.25),
                (float('inf'), 0.30)
            ]
        else:  # old regime
            slabs = [
                (250000, 0),
                (500000, 0.05),
                (1000000, 0.20),
                (float('inf'), 0.30)
            ]

        tax = 0
        prev_limit = 0
        slab_details = []
        for limit, rate in slabs:
            if taxable_income > prev_limit:
                taxable = min(taxable_income, limit) - prev_limit
                slab_tax = taxable * rate
                tax += slab_tax
                slab_details.append({
                    'from': prev_limit,
                    'to': limit,
                    'rate': rate * 100,
                    'amount': taxable,
                    'tax': slab_tax
                })
                prev_limit = limit

        cess = tax * 0.04
        total_tax = tax + cess

        return {
            'total_tax': total_tax,
            'base_tax': tax,
            'cess': cess,
            'effective_rate': (total_tax / income) * 100 if income > 0 else 0,
            'slabs': slab_details,
            'taxable_income': taxable_income
        }

    @staticmethod
    def calculate_gst(amount, category):
        """Calculate GST for different categories with detailed breakdown"""
        rates = {
            'Essential': 0.05,
            'Standard': 0.12,
            'Luxury': 0.18,
            'Special': 0.28
        }
        rate = rates.get(category, 0.12)
        gst_amount = amount * rate
        return {
            'base_price': amount,
            'gst_rate': rate * 100,
            'gst_amount': gst_amount,
            'total': amount + gst_amount,
            'category': category
        }

    @staticmethod
    def calculate_investment_growth(principal, years, rate=0.12, mode='lumpsum', inflation=0.06):
        """Calculate real returns after inflation with year-by-year breakdown"""
        yearly_data = []
        if mode == 'lumpsum':
            nominal = principal
            real = principal
            for year in range(1, years + 1):
                nominal *= (1 + rate)
                real = nominal / ((1 + inflation) ** year)
                yearly_data.append({
                    'year': year,
                    'nominal': nominal,
                    'real': real,
                    'growth': (nominal - principal) / principal * 100
                })
        else:  # SIP mode
            nominal = 0
            for year in range(1, years + 1):
                monthly_investment = principal
                for month in range(1, 13):
                    nominal += monthly_investment
                    nominal *= (1 + rate) ** (1 / 12)
                real = nominal / ((1 + inflation) ** year)
                yearly_data.append({
                    'year': year,
                    'nominal': nominal,
                    'real': real,
                    'growth': (nominal / (principal * 12 * year) - 1) * 100
                })

        final_nominal = yearly_data[-1]['nominal'] if yearly_data else principal
        final_real = yearly_data[-1]['real'] if yearly_data else principal

        return {
            'nominal': final_nominal,
            'real': final_real,
            'yearly': yearly_data,
            'cagr': ((final_nominal / principal) ** (1 / years) - 1) * 100 if years > 0 else 0
        }

    @staticmethod
    def calculate_insurance_needs(age, income, dependents, liabilities, existing_cover=0):
        """Calculate recommended insurance coverage with component breakdown"""
        income_replacement = income * 10  # 10 years income replacement
        liabilities_cover = liabilities * 1.2  # 120% of liabilities
        education_cover = 500000 * dependents  # Rs.5L per dependent for education
        total_needs = income_replacement + liabilities_cover + education_cover
        additional_needed = max(total_needs - existing_cover, 0)

        return {
            'total_needs': total_needs,
            'existing_cover': existing_cover,
            'additional_needed': additional_needed,
            'components': {
                'income_replacement': income_replacement,
                'liabilities_cover': liabilities_cover,
                'education_cover': education_cover
            }
        }

    @staticmethod
    def calculate_opportunity_cost(amount, years, alternative_return=0.12):
        """Calculate what the money could have earned if invested with yearly breakdown"""
        yearly_data = []
        total = 0
        for year in range(1, years + 1):
            total = amount * ((1 + alternative_return) ** year - 1)
            yearly_data.append({
                'year': year,
                'opportunity_cost': total
            })
        return {
            'total': total,
            'yearly': yearly_data
        }

    @staticmethod
    def calculate_financial_health(income, expenses, savings, investments, liabilities, insurance_cover, age):
        """Calculate comprehensive financial health score (0-100) with detailed metrics"""
        # Savings ratio (weight: 25%)
        savings_ratio = savings / income if income > 0 else 0
        savings_score = min(savings_ratio / 0.2, 1) * 25  # 20% is ideal

        # Investment ratio (weight: 25%)
        investment_ratio = investments / (income * max(age - 25, 5)) if income > 0 else 0
        investment_score = min(investment_ratio / 2, 1) * 25  # 2x income by age 35 is good

        # Debt ratio (weight: 20%)
        debt_ratio = liabilities / (income * 0.5) if income > 0 else 0
        debt_score = (1 - min(debt_ratio, 1)) * 20

        # Insurance adequacy (weight: 15%)
        insurance_adequacy = insurance_cover / (income * 10) if income > 0 else 0
        insurance_score = min(insurance_adequacy, 1) * 15

        # Expense ratio (weight: 15%)
        expense_ratio = expenses / income if income > 0 else 0
        excess_expense = max(expense_ratio - 0.6, 0)  # 60% is ideal
        expense_score = (1 - min(excess_expense / 0.4, 1)) * 15  # Cap at 100% expense ratio

        total_score = savings_score + investment_score + debt_score + insurance_score + expense_score

        return {
            'total_score': min(max(total_score, 0), 100),
            'components': {
                'savings': {
                    'ratio': savings_ratio,
                    'score': savings_score,
                    'ideal': 0.2
                },
                'investments': {
                    'ratio': investment_ratio,
                    'score': investment_score,
                    'ideal': 2
                },
                'debt': {
                    'ratio': debt_ratio,
                    'score': debt_score,
                    'ideal': 1.0
                },
                'insurance': {
                    'ratio': insurance_adequacy,
                    'score': insurance_score,
                    'ideal': 1.0
                },
                'expenses': {
                    'ratio': expense_ratio,
                    'score': expense_score,
                    'ideal': 0.6
                }
            }
        }

    @staticmethod
    def get_tax_saving_options(income, regime):
        """Generate tax saving options based on income and regime"""
        options = []

        # Common options for both regimes
        if regime == 'old' or income > 750000:
            options.append(("Section 80C (ELSS, PPF, etc.)", 150000, "Invest in tax-saving instruments"))

        if regime == 'old':
            options.extend([
                ("Section 80D (Health Insurance)", 25000, "Health insurance premium"),
                ("NPS (Additional Rs.50k)", 50000, "National Pension Scheme contribution"),
                ("Home Loan Interest", min(200000, income * 0.3), "Interest on home loan"),
                ("Education Loan Interest", "Full amount", "Interest on education loan"),
                ("Donations (80G)", "Varies", "Eligible charitable donations")
            ])
        else:  # new regime
            options.append(("NPS (Employer Contribution)", 50000, "Employer's NPS contribution"))

        return options

    @staticmethod
    def future_value(amount, years, rate=0.07, inflation=0.03):
        """Calculate real future value considering inflation"""
        nominal = amount * ((1 + rate) ** years)
        real = nominal / ((1 + inflation) ** years)
        return nominal, real

    @staticmethod
    def calculate_value_efficiency(spending_data):
        """Enhanced efficiency calculation with normalization"""
        max_amount = max(data['amount'] for data in spending_data.values())
        return {
            cat: (data['happiness'] / (data['amount'] / max_amount))
            for cat, data in spending_data.items()
        }

    @staticmethod
    def optimize_budget(income, spending_data, min_savings, risk_appetite=0.5):
        """Advanced optimization with risk appetite consideration"""
        prob = LpProblem("BudgetOptimization", LpMaximize)

        # Variables with dynamic bounds based on risk
        vars = {
            cat: LpVariable(
                cat,
                lowBound=data['amount'] * (0.7 if risk_appetite > 0.5 else 0.9),
                upBound=data['amount'] * (1.3 if risk_appetite > 0.5 else 1.1)
            )
            for cat, data in spending_data.items()
        }

        # Objective: Weighted combination of happiness and savings
        prob += sum(
            (data['happiness'] / data['amount']) * vars[cat] * risk_appetite +
            (min_savings / income) * (1 - risk_appetite)
            for cat, data in spending_data.items()
        )

        # Constraints
        prob += sum(vars.values()) <= income - min_savings
        prob += sum(vars.values()) >= income * 0.4  # Minimum spending

        # Category-specific constraints
        for cat, data in spending_data.items():
            if cat.lower() in ['rent', 'utilities']:
                prob += vars[cat] >= data['amount'] * 0.9  # Essential expenses

        prob.solve()

        return {var.name: var.varValue for var in vars.values()}

    @staticmethod
    def bulk_purchase_analysis(unit_price, bulk_price, usage_rate, shelf_life, storage_cost=0):
        """Enhanced bulk analysis with storage costs"""
        break_even = (bulk_price + storage_cost) / unit_price
        return {
            'worthwhile': usage_rate * shelf_life >= break_even,
            'savings': (unit_price * usage_rate * shelf_life) - (bulk_price + storage_cost),
            'break_even': break_even
        }

    @staticmethod
    def generate_spending_forecast(optimized_budget, months=12):
        """Generate future spending forecast with seasonality"""
        forecast = []
        base_values = list(optimized_budget.values())
        categories = list(optimized_budget.keys())

        for month in range(1, months + 1):
            # Add some random variation (5-15%)
            variation = np.random.uniform(0.95, 1.15, len(categories))
            monthly_values = [base * var for base, var in zip(base_values, variation)]

            # Add seasonal effects (e.g., higher shopping in Dec)
            if month == 12:
                seasonal_idx = categories.index('Shopping') if 'Shopping' in categories else -1
                if seasonal_idx >= 0:
                    monthly_values[seasonal_idx] *= 1.5

            forecast.append({
                'Month': (datetime.now() + relativedelta(months=month)).strftime('%b %Y'),
                **dict(zip(categories, monthly_values))
            })

        return pd.DataFrame(forecast)


# ======================
# Business Tools
# ======================
class BusinessTools:
    @staticmethod
    def gst_advisor(turnover, expenses):
        input_tax = sum(e['amount'] * 0.18 for e in expenses if e['gst_eligible'])
        output_tax = turnover * 0.18
        liability = output_tax - input_tax
        comp_scheme_threshold = 1.5 * 1e7
        suggestion = ""
        if turnover < comp_scheme_threshold:
            comp_tax = turnover * 0.01
            if comp_tax < liability:
                suggestion = "Switch to Composition Scheme (1% tax)"
        return {
            "input_tax": input_tax,
            "output_tax": output_tax,
            "liability": liability,
            "suggestion": suggestion
        }

    @staticmethod
    def invoice_discounting(invoice_amount, days_remaining, discount_rate):
        annualized_return = (discount_rate / (1 - discount_rate)) * (365 / days_remaining)
        discounted_amount = invoice_amount * (1 - discount_rate)
        return {
            "discounted_amount": discounted_amount,
            "annualized_return": annualized_return * 100
        }

    @staticmethod
    def working_capital_analysis(inventory_days, receivable_days, payable_days, daily_sales):
        wc_cycle = inventory_days + receivable_days - payable_days
        required_capital = (wc_cycle * daily_sales)
        suggestions = []
        if payable_days < receivable_days:
            suggestions.append("Negotiate longer payment terms with suppliers")
        if inventory_days > 30:
            suggestions.append("Reduce inventory through JIT ordering")
        return {
            "cycle_days": wc_cycle,
            "required_capital": required_capital,
            "suggestions": suggestions
        }

    @staticmethod
    def calculate_clv(avg_order_value, purchase_freq, avg_customer_lifespan, profit_margin=0.25):
        clv = (avg_order_value * purchase_freq * avg_customer_lifespan * profit_margin)
        acquisition_breakeven = clv * 0.3
        return {
            "clv": clv,
            "max_cac": acquisition_breakeven
        }

    @staticmethod
    def vendor_scorecard(price, quality, delivery, payment_terms):
        score = (price * 0.4 + quality * 0.3 + delivery * 0.2 + payment_terms * 0.1)
        grade = "A" if score >= 85 else "B" if score >= 70 else "C"
        return {
            "score": score,
            "grade": grade,
            "improve": "Price" if price < 70 else "Quality" if quality < 80 else None
        }


# ======================
# AI Recommendation Engine
# ======================
class AIAdvisor:
    def __init__(self):
        try:
            self.llm = Ollama(model="llama2")
        except:
            self.llm = OpenAI(temperature=0.7)

    def generate_recommendation(self, user_data, recommendation_type):
        template = """
        Based on the following financial profile:
        Income: {income}
        Age: {age}
        Dependents: {dependents}
        Savings: {savings}
        Investments: {investments}
        Debt: {debt}

        Provide a concise {rec_type} recommendation to improve their financial health.
        Focus on actionable steps and explain why this would help.
        """

        prompt = PromptTemplate(
            input_variables=["income", "age", "dependents", "savings", "investments", "debt", "rec_type"],
            template=template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            income=user_data['income'],
            age=user_data['age'],
            dependents=user_data['dependents'],
            savings=user_data['savings'],
            investments=user_data['investments'],
            debt=user_data['liabilities'],
            rec_type=recommendation_type
        )

        return response


# ======================
# Investment Tracker
# ======================
class InvestmentTracker:
    @staticmethod
    def fetch_live_price(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            return hist['Close'].iloc[-1]
        except:
            return None

    @staticmethod
    def update_portfolio_value(user_id):
        conn = init_db()
        c = conn.cursor()

        # Get all investments
        c.execute("SELECT id, asset_name, quantity FROM investment_portfolio WHERE user_id = ?", (user_id,))
        investments = c.fetchall()

        for inv in investments:
            inv_id, asset_name, quantity = inv
            current_price = InvestmentTracker.fetch_live_price(asset_name)

            if current_price:
                current_value = current_price * quantity
                c.execute("UPDATE investment_portfolio SET current_value = ? WHERE id = ?",
                          (current_value, inv_id))

        conn.commit()

    @staticmethod
    def get_portfolio_performance(user_id):
        conn = init_db()
        c = conn.cursor()

        c.execute('''SELECT asset_type, SUM(current_value) as total_value
                     FROM investment_portfolio
                     WHERE user_id = ?
                     GROUP BY asset_type''', (user_id,))

        portfolio = c.fetchall()
        total_value = sum([x[1] for x in portfolio])

        # Calculate returns
        c.execute('''SELECT SUM(buy_price * quantity) as total_invested
                     FROM investment_portfolio
                     WHERE user_id = ?''', (user_id,))

        total_invested = c.fetchone()[0] or 0
        returns = (total_value - total_invested) / total_invested * 100 if total_invested > 0 else 0

        return {
            'portfolio': portfolio,
            'total_value': total_value,
            'total_invested': total_invested,
            'returns': returns
        }


# ======================
# Enhanced Report Generation with Multi-Year Support
# ======================
class ReportGenerator:
    @staticmethod
    def get_available_report_periods(user_id):
        """Get all available months/years with complete data for the user"""
        conn = init_db()
        query = '''
            SELECT month, year FROM (
                SELECT month, year, COUNT(*) as data_points 
                FROM financial_data 
                WHERE user_id = ?
                GROUP BY month, year
            ) WHERE data_points > 0
            ORDER BY year DESC, month DESC
        '''
        periods = pd.read_sql(query, conn, params=(user_id,))
        conn.close()
        return periods.to_dict('records')

    @staticmethod
    def validate_report_data(data_dict):
        """Validate that we have minimum required data for a report"""
        required_fields = ['financial', 'spending', 'health']
        return all(key in data_dict and not data_dict[key].empty for key in required_fields)

    @staticmethod
    def generate_report_title(month, year):
        """Generate formatted report title based on parameters"""
        if month:
            return f"{calendar.month_name[month]} {year}"
        return f"Annual Report - {year}"

    @staticmethod
    def generate_report_filename(month, year):
        """Generate filename for the report"""
        if month:
            return f"OptiSpend_Report_{calendar.month_name[month]}_{year}.pdf"
        return f"OptiSpend_Annual_Report_{year}.pdf"

    @staticmethod
    def get_comparison_data(user_id, month, year):
        """Get previous period data for comparison"""
        conn = init_db()
        try:
            if month:  # Monthly report
                prev_month = month - 1 if month > 1 else 12
                prev_year = year if month > 1 else year - 1

                prev_data = {
                    'financial': pd.read_sql('''
                        SELECT income, expenses, savings, investments, liabilities
                        FROM financial_data 
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, prev_month, prev_year)),

                    'health': pd.read_sql('''
                        SELECT total_score
                        FROM financial_health
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, prev_month, prev_year))
                }
            else:  # Annual report
                prev_data = {
                    'financial': pd.read_sql('''
                        SELECT SUM(income) as income, SUM(expenses) as expenses, 
                               SUM(savings) as savings, SUM(investments) as investments
                        FROM financial_data 
                        WHERE user_id = ? AND year = ?
                    ''', conn, params=(user_id, year - 1)),

                    'health': pd.read_sql('''
                        SELECT AVG(total_score) as total_score
                        FROM financial_health
                        WHERE user_id = ? AND year = ?
                    ''', conn, params=(user_id, year - 1))
                }

            return prev_data
        finally:
            conn.close()

    @staticmethod
    def generate_report_pdf(data, prev_data, month, year):
        """Generate the PDF report content"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Header
        pdf.set_fill_color(37, 99, 235)  # Blue
        pdf.set_text_color(255, 255, 255)
        title = ReportGenerator.generate_report_title(month, year)
        pdf.cell(0, 10, f"OptiSpend PRO Report - {title}", 0, 1, 'C', 1)
        pdf.ln(10)
        pdf.set_text_color(0, 0, 0)

        # 1. Financial Summary Section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Financial Summary", 0, 1)
        pdf.set_font("Arial", size=12)

        fin_data = data['financial'].iloc[0] if month else data['financial']
        pdf.cell(0, 10, f"Income: Rs. {fin_data['income']:,.0f}", 0, 1)
        pdf.cell(0, 10, f"Expenses: Rs. {fin_data['expenses']:,.0f}", 0, 1)
        pdf.cell(0, 10, f"Savings: Rs. {fin_data['savings']:,.0f}", 0, 1)

        if not prev_data['financial'].empty:
            prev_fin = prev_data['financial'].iloc[0]
            change = fin_data['savings'] - prev_fin.get('savings', 0)
            period = "previous month" if month else "previous year"
            pdf.cell(0, 10, f"Savings Change: Rs. {change:+,.0f} from {period}", 0, 1)
        pdf.ln(5)

        # 2. Financial Health Section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Financial Health", 0, 1)
        pdf.set_font("Arial", size=12)

        health = data['health'].iloc[0] if month else data['health']
        pdf.cell(0, 10, f"Overall Score: {health['total_score']:.0f}/100", 0, 1)

        if not prev_data['health'].empty:
            prev_h = prev_data['health'].iloc[0]
            change = health['total_score'] - prev_h.get('total_score', 0)
            period = "previous month" if month else "previous year"
            pdf.cell(0, 10, f"Change: {change:+.0f} points from {period}", 0, 1)

        if month:  # Only show detailed scores for monthly reports
            pdf.cell(0, 10, f"Savings Score: {health['savings_score']:.0f}/25", 0, 1)
            pdf.cell(0, 10, f"Investment Score: {health['investment_score']:.0f}/25", 0, 1)
            pdf.cell(0, 10, f"Debt Score: {health['debt_score']:.0f}/20", 0, 1)
        pdf.ln(5)

        # 3. Spending Analysis Section (monthly only)
        if month and not data['spending'].empty:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Spending Analysis", 0, 1)
            pdf.set_font("Arial", size=12)

            total_spending = data['spending']['amount'].sum()
            for _, row in data['spending'].iterrows():
                pct = (row['amount'] / total_spending) * 100
                pdf.cell(0, 10,
                         f"{row['category']}: Rs. {row['amount']:,.0f} ({pct:.1f}%) - Happiness: {row['happiness']}/10",
                         0, 1)
            pdf.ln(5)

        # 4. Optimization Results Section (if available)
        if not data['optimizations'].empty:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Budget Optimization", 0, 1)
            pdf.set_font("Arial", size=12)

            total_savings = (
                    data['optimizations']['current_amount'] - data['optimizations']['optimized_amount']).sum()
            pdf.cell(0, 10, f"Total Potential Savings: Rs. {total_savings:,.0f}", 0, 1)

            for _, row in data['optimizations'].iterrows():
                change = ((row['optimized_amount'] - row['current_amount']) / row['current_amount']) * 100
                pdf.cell(0, 10,
                         f"{row['category']}: Rs. {row['current_amount']:,.0f} to Rs. {row['optimized_amount']:,.0f} ({change:+.1f}%)",
                         0, 1)
            pdf.ln(5)

        # 5. Business Tools Usage (if available)
        if not data['business'].empty:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Business Tools Usage", 0, 1)
            pdf.set_font("Arial", size=12)

            for _, row in data['business'].iterrows():
                pdf.cell(0, 10, f"Tool: {row['tool_name']}", 0, 1)
                try:
                    result = json.loads(row['result_data'])
                    if 'liability' in result:
                        pdf.cell(0, 10, f"  GST Liability: Rs. {result['liability']:,.0f}", 0, 1)
                    elif 'discounted_amount' in result:
                        pdf.cell(0, 10, f"  Discounted Amount: Rs. {result['discounted_amount']:,.0f}", 0, 1)
                    elif 'clv' in result:
                        pdf.cell(0, 10, f"  Customer LTV: Rs. {result['clv']:,.0f}", 0, 1)
                except:
                    pass
            pdf.ln(5)

        # 6. Recommendations Section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Recommendations", 0, 1)
        pdf.set_font("Arial", size=12)

        if not data['optimizations'].empty:
            top_increases = data['optimizations'].nlargest(3, 'optimized_amount')
            for _, row in top_increases.iterrows():
                pdf.cell(0, 10, f"- Consider increasing spending on {row['category']}", 0, 1)

            top_decreases = data['optimizations'].nsmallest(3, 'optimized_amount')
            for _, row in top_decreases.iterrows():
                pdf.cell(0, 10, f"- Consider reducing spending on {row['category']}", 0, 1)

        if health['savings_score'] < 20:
            pdf.cell(0, 10, "- Increase your savings rate to at least 20% of income", 0, 1)
        if 'insurance_score' in health and health['insurance_score'] < 12:
            pdf.cell(0, 10, "- Review your insurance coverage for adequacy", 0, 1)

        return pdf.output(dest='S').encode('latin1')

    @staticmethod
    def generate_monthly_report(user_id, month, year, return_data=False):
        """Generate comprehensive monthly report"""
        conn = None
        try:
            conn = init_db()
            data = {}

            # Get all data in a single transaction
            with conn:
                data = {
                    'financial': pd.read_sql('''
                        SELECT * FROM financial_data 
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, month, year)),

                    'spending': pd.read_sql('''
                        SELECT category, amount, happiness, essential 
                        FROM spending_data 
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, month, year)),

                    'optimizations': pd.read_sql('''
                        SELECT category, current_amount, optimized_amount, happiness
                        FROM optimizations
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, month, year)),

                    'health': pd.read_sql('''
                        SELECT total_score, savings_score, investment_score,
                               debt_score, insurance_score, expense_score
                        FROM financial_health
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, month, year)),

                    'business': pd.read_sql('''
                        SELECT tool_name, input_data, result_data
                        FROM business_data
                        WHERE user_id = ? AND month = ? AND year = ?
                    ''', conn, params=(user_id, month, year))
                }

                if not ReportGenerator.validate_report_data(data):
                    raise ValueError(f"Incomplete data for {calendar.month_name[month]} {year}")

            # Get previous month data for comparison
            prev_data = ReportGenerator.get_comparison_data(user_id, month, year)

            # Generate PDF
            pdf_bytes = ReportGenerator.generate_report_pdf(data, prev_data, month, year)

            if return_data:
                return pdf_bytes, data
            return pdf_bytes

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()

    @staticmethod
    def generate_annual_report(user_id, year):
        """Generate comprehensive annual report"""
        conn = None
        try:
            conn = init_db()
            data = {}

            # Get all data in a single transaction
            with conn:
                data = {
                    'financial': pd.read_sql('''
                        SELECT SUM(income) as income, SUM(expenses) as expenses, 
                               SUM(savings) as savings, SUM(investments) as investments
                        FROM financial_data 
                        WHERE user_id = ? AND year = ?
                    ''', conn, params=(user_id, year)),

                    'optimizations': pd.read_sql('''
                        SELECT category, SUM(current_amount) as current_amount, 
                               SUM(optimized_amount) as optimized_amount, AVG(happiness) as happiness
                        FROM optimizations
                        WHERE user_id = ? AND year = ?
                        GROUP BY category
                    ''', conn, params=(user_id, year)),

                    'health': pd.read_sql('''
                        SELECT AVG(total_score) as total_score, AVG(savings_score) as savings_score,
                               AVG(investment_score) as investment_score, AVG(debt_score) as debt_score
                        FROM financial_health
                        WHERE user_id = ? AND year = ?
                    ''', conn, params=(user_id, year)),

                    'business': pd.read_sql('''
                        SELECT tool_name, COUNT(*) as usage_count
                        FROM business_data
                        WHERE user_id = ? AND year = ?
                        GROUP BY tool_name
                    ''', conn, params=(user_id, year))
                }

            # Get previous year data for comparison
            prev_data = ReportGenerator.get_comparison_data(user_id, None, year)

            # Generate PDF
            pdf_bytes = ReportGenerator.generate_report_pdf(data, prev_data, None, year)
            return pdf_bytes

        except Exception as e:
            st.error(f"Error generating annual report: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()

    @staticmethod
    def show_report_interface(user_id):
        """Enhanced UI for report generation with annual reports"""
        st.header("📄 Financial Reports")

        available_periods = ReportGenerator.get_available_report_periods(user_id)
        if not available_periods:
            st.warning("No complete report data available yet.")
            return

        # Get unique years from available data
        available_years = sorted(list({p['year'] for p in available_periods}), reverse=True)

        # Create tabs for different report types
        tab1, tab2 = st.tabs(["📅 Monthly Reports", "📊 Annual Reports"])

        with tab1:
            st.subheader("Monthly Reports")

            # Group months by year for better organization
            for year in available_years:
                with st.expander(f"Year {year}"):
                    months_in_year = [p for p in available_periods if p['year'] == year]
                    for period in months_in_year:
                        month_name = calendar.month_name[period['month']]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{month_name} {year}")
                        with col2:
                            if st.button("Generate", key=f"gen_month_{period['month']}_{period['year']}"):
                                with st.spinner(f"Generating {month_name} {year} report..."):
                                    report_bytes = ReportGenerator.generate_monthly_report(
                                        user_id, period['month'], period['year'])

                                    if report_bytes:
                                        filename = ReportGenerator.generate_report_filename(
                                            period['month'], period['year'])

                                        st.download_button(
                                            label="Download Report",
                                            data=report_bytes,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"dl_month_{period['month']}_{period['year']}"
                                        )

        with tab2:
            st.subheader("Annual Reports")

            for year in available_years:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"Year {year}")
                with col2:
                    if st.button("Generate", key=f"gen_year_{year}"):
                        with st.spinner(f"Generating annual report for {year}..."):
                            report_bytes = ReportGenerator.generate_annual_report(user_id, year)

                            if report_bytes:
                                filename = ReportGenerator.generate_report_filename(None, year)

                                st.download_button(
                                    label="Download Report",
                                    data=report_bytes,
                                    file_name=filename,
                                    mime="application/pdf",
                                    key=f"dl_year_{year}"
                                )

    @staticmethod
    def show_sidebar_report_options(user_id):
        """Show simplified report options in the sidebar"""
        st.sidebar.header("Quick Reports")

        available_periods = ReportGenerator.get_available_report_periods(user_id)
        if not available_periods:
            st.sidebar.warning("No report data available")
            return

        # Recent months
        st.sidebar.subheader("Recent Months")
        recent_months = available_periods[:3]  # Get last 3 months
        for period in recent_months:
            month_name = calendar.month_name[period['month']]
            if st.sidebar.button(f"{month_name} {period['year']}",
                                 key=f"sidebar_month_{period['month']}_{period['year']}"):
                with st.spinner(f"Generating {month_name} report..."):
                    report_bytes = ReportGenerator.generate_monthly_report(
                        user_id, period['month'], period['year'])

                    if report_bytes:
                        filename = ReportGenerator.generate_report_filename(
                            period['month'], period['year'])

                        st.sidebar.download_button(
                            label="Download Report",
                            data=report_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"sidebar_dl_month_{period['month']}_{period['year']}"
                        )

            # Annual report
            st.sidebar.subheader("Annual Report")
            # Get unique years from available data
            available_years = sorted(list({p['year'] for p in available_periods}), reverse=True)

            selectbox_key = f"sidebar_year_select_{user_id}"
            selected_year = st.sidebar.selectbox(
                "Select Year",
                available_years,
                key=selectbox_key
            )

            generate_key = f"sidebar_gen_annual_{user_id}"
            if st.sidebar.button("Generate Annual Report", key=generate_key):
                with st.spinner(f"Generating {selected_year} annual report..."):
                    report_bytes = ReportGenerator.generate_annual_report(user_id, selected_year)

                    if report_bytes:
                        filename = ReportGenerator.generate_report_filename(None, selected_year)

                        download_key = f"sidebar_dl_year_{selected_year}_{user_id}"
                        st.sidebar.download_button(
                            label="Download Annual Report",
                            data=report_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key=download_key
                        )

# ======================
# UI Components
# ======================
class FinancialDashboard:
    @staticmethod
    def show_main_dashboard():
        st.header("📊 Financial Overview")

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual Income", f"Rs.{st.session_state.financial_data['income']:,.0f}")

        tax_data = FinancialCalculator.calculate_income_tax(
            st.session_state.financial_data['income'],
            st.session_state.financial_data['tax_regime']
        )
        m2.metric("Tax Liability",
                  f"Rs.{tax_data['total_tax']:,.0f}",
                  f"{tax_data['effective_rate']:.1f}% effective rate")

        m3.metric("Savings Rate",
                  f"{(st.session_state.financial_data['savings'] / st.session_state.financial_data['income']) * 100:.1f}%",
                  f"Rs.{st.session_state.financial_data['savings']:,.0f}")

        insurance_adequacy = st.session_state.financial_data['insurance_cover'] / (
                st.session_state.financial_data['income'] * 10)
        m4.metric("Insurance Adequacy",
                  f"{min(insurance_adequacy * 100, 100):.0f}%",
                  f"Rs.{st.session_state.financial_data['insurance_cover']:,.0f} cover")

        # Expense analysis
        st.subheader("💸 Expense Breakdown")
        FinancialDashboard.show_expense_analysis()

        # Opportunity cost visualization
        st.subheader("⏳ Opportunity Cost Analysis")
        FinancialDashboard.show_opportunity_cost_analysis()

    @staticmethod
    def show_expense_analysis():
        expense_categories = {
            'Housing': 0.25,
            'Food': 0.15,
            'Transport': 0.10,
            'Insurance': 0.08,
            'Entertainment': 0.10,
            'Shopping': 0.10,
            'Utilities': 0.08,
            'Healthcare': 0.07,
            'Education': 0.05,
            'Other': 0.10
        }
        expenses = {
            cat: st.session_state.financial_data['expenses'] * pct
            for cat, pct in expense_categories.items()
        }

        fig = px.pie(
            names=list(expenses.keys()),
            values=list(expenses.values()),
            hole=0.4,
            title="Expense Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show GST impact
        gst_rates = {
            'Housing': 'Standard',
            'Food': 'Essential',
            'Transport': 'Standard',
            'Insurance': 'Standard',
            'Entertainment': 'Luxury',
            'Shopping': 'Luxury',
            'Utilities': 'Essential',
            'Healthcare': 'Standard',
            'Education': 'Standard',
            'Other': 'Standard'
        }
        gst_impact = {
            cat: FinancialCalculator.calculate_gst(amt, gst_rates[cat])['gst_amount']
            for cat, amt in expenses.items()
        }

        fig = px.bar(
            x=list(gst_impact.keys()),
            y=list(gst_impact.values()),
            title="Annual GST Impact by Category",
            labels={'x': 'Category', 'y': 'GST Amount (Rs.)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def show_opportunity_cost_analysis():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Recurring Expenses")
            recurring_expenses = {
                'Dining Out': 3000,
                'Streaming Subscriptions': 1000,
                'Gym Membership': 1500,
                'Cigarettes/Alcohol': 2000
            }

            years = st.slider("Time Period (years)", 1, 30, 5, key='recurring_years')

            opp_cost_data = []
            for name, amount in recurring_expenses.items():
                annual = amount * 12
                cost = FinancialCalculator.calculate_opportunity_cost(annual, years)
                opp_cost_data.append({
                    'Expense': name,
                    'Monthly': amount,
                    'Opportunity Cost': cost['total']
                })

            df = pd.DataFrame(opp_cost_data)
            fig = px.bar(
                df,
                x='Expense',
                y='Opportunity Cost',
                hover_data=['Monthly'],
                title=f"What you could have in {years} years if invested",
                labels={'Opportunity Cost': 'Potential Value (Rs.)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### One-Time Purchases")
            purchases = {
                'Luxury Watch': 50000,
                'International Vacation': 200000,
                'New Smartphone': 80000
            }

            years = st.slider("Time Period (years)", 1, 30, 5, key='purchase_years')

            opp_cost_data = []
            for name, amount in purchases.items():
                cost = FinancialCalculator.calculate_opportunity_cost(amount, years)
                opp_cost_data.append({
                    'Purchase': name,
                    'Amount': amount,
                    'Opportunity Cost': cost['total']
                })

            df = pd.DataFrame(opp_cost_data)
            fig = px.bar(
                df,
                x='Purchase',
                y='Opportunity Cost',
                hover_data=['Amount'],
                title=f"Potential growth if invested for {years} years",
                labels={'Opportunity Cost': 'Potential Value (Rs.)'}
            )
            st.plotly_chart(fig, use_container_width=True)


class BudgetOptimizer:
    @staticmethod
    def show_optimization():
        st.header("💰 Budget Optimization")

        if st.button("Run Optimization", type="primary"):
            st.session_state.optimized_budget = FinancialCalculator.optimize_budget(
                st.session_state.financial_data['income'],
                st.session_state.spending_data,
                st.session_state.min_savings,
                st.session_state.risk_appetite
            )

            # Save optimization results
            user_id = get_user_id()
            save_optimization_results(user_id, st.session_state.spending_data, st.session_state.optimized_budget)

        if st.session_state.get('optimized_budget'):
            # Optimization results
            st.subheader("Optimized Budget Allocation")

            # Create comparison dataframe
            comparison = pd.DataFrame({
                'Category': list(st.session_state.spending_data.keys()),
                'Current': [data['amount'] for data in st.session_state.spending_data.values()],
                'Optimized': [st.session_state.optimized_budget[cat] for cat in st.session_state.spending_data.keys()],
                'Happiness': [data['happiness'] for data in st.session_state.spending_data.values()]
            })

            comparison['Change'] = (comparison['Optimized'] - comparison['Current']) / comparison['Current']
            comparison['Change Amount'] = comparison['Optimized'] - comparison['Current']

            # Show summary metrics
            current_total = comparison['Current'].sum()
            optimized_total = comparison['Optimized'].sum()
            current_happiness = (comparison['Current'] * comparison['Happiness']).sum() / current_total
            optimized_happiness = (comparison['Optimized'] * comparison['Happiness']).sum() / optimized_total

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Spending",
                      f"Rs.{optimized_total:,.0f}",
                      f"{optimized_total - current_total:+,.0f} vs current")
            m2.metric("Savings",
                      f"Rs.{st.session_state.financial_data['income'] - optimized_total:,.0f}",
                      f"{(st.session_state.financial_data['income'] - optimized_total) / st.session_state.financial_data['income']:.0%} of income")
            m3.metric("Avg Happiness",
                      f"{optimized_happiness:.1f}/10",
                      f"{optimized_happiness - current_happiness:+.1f} change")

            # Show detailed changes
            st.subheader("Detailed Changes")
            fig = px.bar(
                comparison.melt(id_vars=['Category', 'Happiness'],
                                value_vars=['Current', 'Optimized']),
                x='Category',
                y='value',
                color='variable',
                barmode='group',
                hover_data=['Happiness'],
                color_discrete_map={
                    'Current': '#636EFA',
                    'Optimized': '#00CC96'
                },
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show biggest changes
            st.subheader("Biggest Improvements")
            top_increases = comparison.nlargest(3, 'Change')
            top_decreases = comparison.nsmallest(3, 'Change')

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 💹 Recommended Increases")
                for _, row in top_increases.iterrows():
                    st.markdown(f"""
                    <div class="card">
                        <h4>{row['Category']}</h4>
                        <p>Increase by <span class="highlight">+Rs.{row['Change Amount']:,.0f}</span> ({row['Change']:+.0%})</p>
                        <p>Happiness: {row['Happiness']}/10</p>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                st.markdown("##### 📉 Recommended Decreases")
                for _, row in top_decreases.iterrows():
                    st.markdown(f"""
                    <div class="card">
                        <h4>{row['Category']}</h4>
                        <p>Decrease by <span class="highlight">Rs.{row['Change Amount']:,.0f}</span> ({row['Change']:+.0%})</p>
                        <p>Happiness: {row['Happiness']}/10</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Optimization' to generate recommendations")


class ForecastPlanner:
    @staticmethod
    def show_forecast():
        st.header("🔮 Financial Forecast")

        if st.session_state.get('optimized_budget'):
            forecast_months = st.slider("Forecast Period (months)", 6, 60, 12)
            forecast = FinancialCalculator.generate_spending_forecast(st.session_state.optimized_budget,
                                                                      forecast_months)

            # Show forecast
            st.subheader("Spending Forecast")
            fig = px.line(
                forecast.melt(id_vars='Month'),
                x='Month',
                y='value',
                color='variable',
                title="Projected Monthly Spending",
                labels={'value': 'Amount (Rs.)', 'variable': 'Category'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show cumulative savings
            st.subheader("Savings Potential")
            monthly_savings = st.session_state.financial_data['income'] - forecast.drop('Month', axis=1).sum(axis=1)
            cumulative_savings = monthly_savings.cumsum()

            fig = px.area(
                x=forecast['Month'],
                y=cumulative_savings,
                title="Cumulative Savings Over Time",
                labels={'x': 'Month', 'y': 'Savings (Rs.)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Calculate investment growth
            nominal, real = FinancialCalculator.future_value(cumulative_savings.iloc[-1] / forecast_months * 12,
                                                             forecast_months / 12)
            st.metric(
                "Potential Investment Value",
                f"Rs.{nominal:,.0f}",
                f"Rs.{real:,.0f} in today's money (after inflation)"
            )
        else:
            st.warning("Run optimization first to generate forecasts")


class AIAdvisorUI:
    @staticmethod
    def show_advisor():
        st.header("🧠 AI Financial Advisor")

        # Bulk purchase analyzer
        st.subheader("Bulk Purchase Advisor")
        with st.form("bulk_form"):
            c1, c2, c3, c4 = st.columns(4)
            unit_price = c1.number_input("Unit Price (Rs.)", min_value=1, value=50)
            bulk_price = c2.number_input("Bulk Price (Rs.)", min_value=1, value=400)
            usage_rate = c3.number_input("Usage per month", min_value=1, value=4)
            shelf_life = c4.number_input("Shelf Life (months)", min_value=1, value=6)

            submitted = st.form_submit_button("Analyze")

            if submitted:
                analysis = FinancialCalculator.bulk_purchase_analysis(unit_price, bulk_price, usage_rate, shelf_life)

                if analysis['worthwhile']:
                    st.success(f""" 
                    **✅ Recommended Purchase**
                    - Potential savings: Rs.{analysis['savings']:,.0f} over {shelf_life} months
                    - Break-even point: {analysis['break_even']:.1f} units
                    """)
                else:
                    st.error(f"""
                    **❌ Not Recommended**
                    - You would need to use {analysis['break_even']:.1f} units to break even
                    - At current usage, you'll only use {usage_rate * shelf_life} units
                    """)
        # Opportunity cost calculator
        st.subheader("Opportunity Cost Calculator")
        with st.form("opp_cost_form"):
            c1, c2 = st.columns(2)
            expense = c1.selectbox(
                "Select Expense",
                list(st.session_state.spending_data.keys())
            )
            amount = c2.number_input(
                "Amount (Rs.)",
                value=st.session_state.spending_data[expense]['amount'],
                min_value=100
            )

            years = st.slider("Time Horizon (years)", 1, 30, 5)

            submitted = st.form_submit_button("Calculate")

            if submitted:
                nominal, real = FinancialCalculator.future_value(amount * 12, years)

                st.markdown(f"""
                <div class="card">
                    <h4>If you invested Rs.{amount:,.0f}/month instead...</h4>
                    <p>After {years} years you'd have:</p>
                    <h3>Rs.{nominal:,.0f}</h3>
                    <p>(Rs.{real:,.0f} in today's purchasing power)</p>
                    <p>Current happiness from this expense: {st.session_state.spending_data[expense]['happiness']}/10</p>
                </div>
                """, unsafe_allow_html=True)

                # Show comparison points
                st.markdown("**Equivalent to:**")
                comparisons = [
                    (200000, "years of rent at Rs.20k/month"),
                    (50000, "international vacations"),
                    (10000, "months of grocery bills"),
                    (15000, "new smartphones")
                ]

                for value, desc in comparisons:
                    st.write(f"- {nominal / value:.1f} {desc}")


class TaxCenter:
    @staticmethod
    def show_tax_center():
        st.header("🧾 Tax Optimization Center")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Income Tax Planner")

            deductions = st.number_input(
                "Available Deductions (80C, HRA, etc.) (Rs.)",
                min_value=0,
                value=150000,
                step=10000,
                key="tax_deductions"
            )

            tax_data = FinancialCalculator.calculate_income_tax(
                st.session_state.financial_data['income'],
                st.session_state.financial_data['tax_regime'],
                deductions
            )

            st.markdown(f"""
            <div class="card tax-card">
                <h3>Tax Liability: Rs.{tax_data['total_tax']:,.0f}</h3>
                <p>Effective Rate: {tax_data['effective_rate']:.1f}%</p>
                <p>Tax Savings from Deductions: Rs.{FinancialCalculator.calculate_income_tax(st.session_state.financial_data['income'], st.session_state.financial_data['tax_regime'])['total_tax'] - tax_data['total_tax']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Visual tax slab breakdown
            st.markdown("### Tax Slab Breakdown")
            slab_data = tax_data['slabs']

            # Create DataFrame with proper column names
            df = pd.DataFrame(slab_data)
            if not df.empty:
                # Ensure the DataFrame has the expected columns
                if 'rate' in df.columns and 'tax' in df.columns:
                    fig = px.bar(
                        df,
                        x='rate',
                        y='tax',
                        title="Tax by Slab",
                        labels={'rate': 'Tax Rate (%)', 'tax': 'Tax Amount (Rs.)'},
                        hover_data=['from', 'to', 'amount']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Tax slab data is not in the expected format")
            else:
                st.warning("No tax slab data available")

            # Tax regime comparison
            new_tax = \
                FinancialCalculator.calculate_income_tax(st.session_state.financial_data['income'], 'new', deductions)[
                    'total_tax']
            old_tax = \
                FinancialCalculator.calculate_income_tax(st.session_state.financial_data['income'], 'old', deductions)[
                    'total_tax']
            better = "New" if new_tax < old_tax else "Old"

            st.markdown(f"""
            <div class="card">
                <h4>Regime Comparison</h4>
                <p>New Regime: Rs.{new_tax:,.0f}</p>
                <p>Old Regime: Rs.{old_tax:,.0f}</p>
                <p class="highlight">{better} regime saves you Rs.{abs(new_tax - old_tax):,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### GST Optimizer")

            with st.form("gst_optimizer"):
                purchase_amount = st.number_input("Purchase Amount (Rs.)", min_value=0, value=50000,
                                                  key="gst_purchase_amount")
                category = st.selectbox(
                    "Category",
                    ['Essential', 'Standard', 'Luxury', 'Special'],
                    index=1,
                    key="gst_category"
                )

                if st.form_submit_button("Calculate GST Impact"):
                    gst_data = FinancialCalculator.calculate_gst(purchase_amount, category)

                    st.markdown(f"""
                    <div class="card tax-card">
                        <h4>GST Breakdown</h4>
                        <p>Base Price: Rs.{gst_data['base_price']:,.0f}</p>
                        <p>GST ({gst_data['category']} @ {gst_data['gst_rate']:.0f}%): Rs.{gst_data['gst_amount']:,.0f}</p>
                        <hr>
                        <h3>Total: Rs.{gst_data['total']:,.0f}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Business expense consideration
                    if st.checkbox("Is this a business expense?", key="business_expense"):
                        tax_saving = gst_data['gst_amount'] * 0.30  # Assuming 30% tax bracket
                        st.markdown(f"""
                        <div class="card">
                            <h4>Tax Benefit</h4>
                            <p>As business expense, you may save:</p>
                            <p>Income Tax: Rs.{tax_saving:,.0f}</p>
                            <p>Effective Cost: Rs.{gst_data['total'] - tax_saving:,.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # Tax saving opportunities
            st.markdown("### Tax Saving Opportunities")
            tax_options = FinancialCalculator.get_tax_saving_options(
                st.session_state.financial_data['income'],
                st.session_state.financial_data['tax_regime']
            )

            for option, amount, description in tax_options:
                st.markdown(f"""
                <div class="card">
                    <h4>{option}</h4>
                    <p>Limit: Rs.{amount if isinstance(amount, int) else amount}</p>
                    <p><small>{description}</small></p>
                </div>
                """, unsafe_allow_html=True)


class InsurancePlanner:
    @staticmethod
    def show_insurance_analysis():
        st.header("🛡️ Insurance Planning")

        col1, col2 = st.columns(2)

        with col1:
            health_expander = st.expander("🏥 Health Insurance", expanded=True)
            with health_expander:
                health_cover = st.number_input("Current Health Cover (Rs.)", min_value=0, value=500000, step=10000,
                                               key="health_cover")
                health_premium = st.number_input("Annual Premium (Rs.)", min_value=0, value=15000, step=1000,
                                                 key="health_premium")
                family_members = st.number_input("Family Members Covered", min_value=1, value=3, key="family_members")

                st.markdown(f"""
                <div class="card insurance-card">
                    <h4>Health Insurance Analysis</h4>
                    <p>Recommended Cover: Rs.{max(500000 * family_members, 1000000):,.0f}</p>
                    <p>Current Adequacy: {"✅ Sufficient" if health_cover >= 500000 * family_members else "⚠️ Insufficient"}</p>
                    <p>Premium Efficiency: Rs.{health_cover / health_premium:,.0f} cover per Rs. premium</p>
                </div>
                """, unsafe_allow_html=True)

                # Opportunity cost of premium
                years = st.slider("Time Horizon for Analysis (years)", 1, 30, 10, key="health_opp_years")
                opp_cost = FinancialCalculator.calculate_opportunity_cost(health_premium, years)
                st.markdown(f"**Opportunity Cost:** Rs.{opp_cost['total']:,.0f} potential returns if invested instead")

        with col2:
            life_expander = st.expander("🧑‍🤝‍🧑 Life Insurance", expanded=True)
            with life_expander:
                life_cover = st.number_input("Current Life Cover (₹)", min_value=0, value=1000000, step=100000,
                                             key="life_cover")
                life_premium = st.number_input("Annual Premium (₹)", min_value=0, value=25000, step=1000,
                                               key="life_premium")
                age = st.number_input("Your Age", min_value=18, max_value=70, value=30, key="life_age")
                dependents = st.number_input("Number of Dependents", min_value=0, value=2, key="life_dependents")
                liabilities = st.number_input("Total Liabilities (₹)", min_value=0, value=500000, step=100000,
                                              key="life_liabilities")

                insurance_needs = FinancialCalculator.calculate_insurance_needs(
                    age, st.session_state.financial_data['income'],
                    dependents, liabilities, life_cover
                )

                st.markdown(f"""
                <div class="card insurance-card">
                    <h4>Life Insurance Analysis</h4>
                    <p>Recommended Additional Cover: ₹{insurance_needs['additional_needed']:,.0f}</p>
                    <p>Current Adequacy: {"✅ Sufficient" if life_cover >= insurance_needs['total_needs'] * 0.8 else "⚠️ Insufficient"}</p>
                    <p>Premium Efficiency: ₹{life_cover / life_premium:,.0f} cover per ₹ premium</p>
                </div>
                """, unsafe_allow_html=True)

                # Term vs Whole Life comparison
                term_cost = st.session_state.financial_data['income'] * 0.01
                whole_life_cost = term_cost * 3
                cost_diff = whole_life_cost - term_cost
                opp_cost_diff = FinancialCalculator.calculate_opportunity_cost(cost_diff, 20)['total']

                st.markdown(f"""
                <div class="card">
                    <h4>Insurance Type Comparison</h4>
                    <p>Term Insurance (₹{term_cost:,.0f}/yr): Pure protection</p>
                    <p>Whole Life (₹{whole_life_cost:,.0f}/yr): Protection + Investment</p>
                    <p class="highlight">Opportunity Cost Difference: ₹{opp_cost_diff:,.0f} over 20 years</p>
                </div>
                """, unsafe_allow_html=True)

                # Visual breakdown of insurance needs
                fig = px.pie(
                    names=list(insurance_needs['components'].keys()),
                    values=list(insurance_needs['components'].values()),
                    title="Life Insurance Needs Breakdown",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)


class InvestmentPlanner:
    @staticmethod
    def show_investment_analysis():
        st.header("📈 Investment Planning")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Investment Growth Calculator")

            with st.form("investment_calculator"):
                principal = st.number_input(
                    "Investment Amount (₹)",
                    min_value=0,
                    value=100000,
                    step=10000,
                    key="investment_principal"
                )

                years = st.slider("Investment Horizon (years)", 1, 30, 10, key="investment_years")
                rate = st.slider(
                    "Expected Annual Return (%)",
                    1.0, 30.0, 12.0,
                    step=0.5,
                    key="investment_rate"
                ) / 100

                mode = st.radio(
                    "Investment Mode",
                    ['lumpsum', 'sip'],
                    format_func=lambda x: "Lump Sum" if x == 'lumpsum' else "Monthly SIP",
                    key="investment_mode"
                )

                if st.form_submit_button("Calculate Projection"):
                    investment_data = FinancialCalculator.calculate_investment_growth(
                        principal, years, rate, mode
                    )

                    st.markdown(f"""
                    <div class="card investment-card">
                        <h4>Investment Projection</h4>
                        <p>After {years} years at {rate * 100:.1f}% return:</p>
                        <h3>Nominal Value: ₹{investment_data['nominal']:,.0f}</h3>
                        <h3>Real Value (after inflation): ₹{investment_data['real']:,.0f}</h3>
                        <p>CAGR: {investment_data['cagr']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Year-by-year growth chart
                    df = pd.DataFrame(investment_data['yearly'])
                    fig = px.line(
                        df,
                        x='year',
                        y=['nominal', 'real'],
                        title="Investment Growth Over Time",
                        labels={'value': 'Value (₹)', 'year': 'Year'},
                        color_discrete_map={'nominal': '#7C3AED', 'real': '#059669'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Portfolio Optimizer")

            current_portfolio = {
                'Equity': 50,
                'Debt': 30,
                'Gold': 10,
                'International': 5,
                'Cash': 5
            }

            fig = px.pie(
                names=list(current_portfolio.keys()),
                values=list(current_portfolio.values()),
                hole=0.4,
                title="Current Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recommended allocation based on age
            equity_percent = max(100 - st.session_state.financial_data['age'], 40)
            recommended = {
                'Equity': equity_percent,
                'Debt': min(40, 100 - equity_percent - 10),
                'Gold': 10,
                'International': 5,
                'Cash': 5
            }

            st.markdown(f"""
            <div class="card">
                <h4>Recommended Allocation (Age {st.session_state.financial_data['age']})</h4>
                <p>Equity: {equity_percent}%</p>
                <p>Debt: {recommended['Debt']}%</p>
                <p>Gold: 10%</p>
                <p>International: 5%</p>
            </div>
            """, unsafe_allow_html=True)

            # Visual comparison
            df = pd.DataFrame({
                'Type': ['Current', 'Recommended'],
                'Equity': [current_portfolio['Equity'], equity_percent],
                'Debt': [current_portfolio['Debt'], recommended['Debt']],
                'Gold': [current_portfolio['Gold'], 10],
                'International': [current_portfolio['International'], 5]
            })
            fig = px.bar(
                df,
                x='Type',
                y=['Equity', 'Debt', 'Gold', 'International'],
                title="Current vs Recommended Allocation",
                labels={'value': 'Percentage (%)'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tax-saving investments
            st.markdown("### Tax-Saving Options")
            tax_options = {
                'Section 80C (ELSS/PPF)': 150000,
                'NPS (Additional ₹50k)': 50000,
                'Health Insurance': 25000,
                'Home Loan Interest': 200000
            }

            df = pd.DataFrame.from_dict({
                'Option': list(tax_options.keys()),
                'Amount': list(tax_options.values())
            })
            st.dataframe(df, hide_index=True)


class FinancialHealthAnalyzer:
    @staticmethod
    def show_health_analysis():
        st.header("🏆 Financial Health Analysis")

        # Calculate health score with detailed breakdown
        health_params = {
            'income': st.session_state.financial_data['income'],
            'expenses': st.session_state.financial_data['expenses'],
            'savings': st.session_state.financial_data['savings'],
            'investments': st.session_state.financial_data['investments'],
            'liabilities': st.session_state.financial_data['liabilities'],
            'insurance_cover': st.session_state.financial_data['insurance_cover'],
            'age': st.session_state.financial_data['age']
        }
        health_data = FinancialCalculator.calculate_financial_health(**health_params)
        health_score = health_data['total_score']

        # Save health score to database
        user_id = get_user_id()
        save_health_score(user_id, health_data)
        update_leaderboard(user_id, health_data['total_score'])
        # Health score visualization
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card health-card">
                <h2>Your Financial Health Score: {health_score:.0f}/100</h2>
                <p>{'Excellent' if health_score >= 80 else 'Good' if health_score >= 60 else 'Fair' if health_score >= 40 else 'Needs Improvement'}</p>
                <div style="height: 10px; background: #E5E7EB; border-radius: 10px; margin: 1rem 0;">
                    <div style="width: {health_score}%; height: 100%; background: {'#059669' if health_score >= 60 else '#F59E0B' if health_score >= 40 else '#DC2626'}; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Spider chart for health components
            components = health_data['components']
            df = pd.DataFrame({
                'Metric': ['Savings', 'Investments', 'Debt', 'Insurance', 'Expenses'],
                'Score': [
                    components['savings']['score'] / 25 * 100,
                    components['investments']['score'] / 25 * 100,
                    components['debt']['score'] / 20 * 100,
                    components['insurance']['score'] / 15 * 100,
                    components['expenses']['score'] / 15 * 100
                ],
                'Ideal': [100, 100, 100, 100, 100]
            })

            fig = px.line_polar(
                df,
                r='Score',
                theta='Metric',
                line_close=True,
                color_discrete_sequence=['#2563EB'],
                title="Financial Health Components"
            )
            fig.add_trace(px.line_polar(
                df,
                r='Ideal',
                theta='Metric',
                line_close=True
            ).data[0])
            fig.update_traces(fill='toself', opacity=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Key metrics with visual indicators
            st.markdown("### Key Metrics")
            metrics = [
                ("Savings Rate",
                 f"{(health_params['savings'] / health_params['income']) * 100:.1f}%",
                 "20%+ is good",
                 components['savings']['ratio'] / components['savings']['ideal']),
                ("Investment Ratio",
                 f"{(health_params['investments'] / health_params['income']):.1f}x income",
                 f"{max(health_params['age'] - 25, 5):.1f}x is good",
                 components['investments']['ratio'] / components['investments']['ideal']),
                ("Debt Ratio",
                 f"{(health_params['liabilities'] / (health_params['income'] * 0.5)) if health_params['income'] > 0 else 0:.1f}",
                 "Below 1.0 is good",
                 1 - components['debt']['ratio'] / components['debt']['ideal']),
                ("Insurance Cover",
                 f"{(health_params['insurance_cover'] / health_params['income']):.1f}x income",
                 "10x is recommended",
                 components['insurance']['ratio'] / components['insurance']['ideal'])
            ]

            for name, value, target, progress in metrics:
                st.markdown(f"""
                <div class="card">
                    <p><strong>{name}</strong>: {value}</p>
                    <p><small>Target: {target}</small></p>
                    <div style="height: 5px; background: #E5E7EB; border-radius: 5px; margin: 0.5rem 0;">
                        <div style="width: {min(max(progress, 0), 1) * 100}%; height: 100%; background: {'#059669' if progress >= 0.8 else '#F59E0B' if progress >= 0.5 else '#DC2626'}; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Improvement recommendations
            st.markdown("### Improvement Recommendations")
            recommendations = []

            # Savings recommendations
            savings_ratio = health_params['savings'] / health_params['income']
            if savings_ratio < 0.2:
                savings_gap = (0.2 - savings_ratio) * health_params['income']
                recommendations.append((
                    "Increase Savings",
                    f"₹{savings_gap:,.0f}/year needed to reach 20% savings rate",
                    "Set up automatic transfers to savings account"
                ))

            # Investment recommendations
            investment_ratio = health_params['investments'] / health_params['income']
            target_ratio = max(health_params['age'] - 25, 5)
            if investment_ratio < target_ratio:
                investment_gap = (target_ratio - investment_ratio) * health_params['income']
                recommendations.append((
                    "Boost Investments",
                    f"₹{investment_gap:,.0f} needed to reach {target_ratio:.1f}x income",
                    "Increase SIP amounts or start new investments"
                ))

            # Debt recommendations
            debt_ratio = health_params['liabilities'] / (health_params['income'] * 0.5)
            if debt_ratio > 1.0:
                recommendations.append((
                    "Reduce Debt",
                    f"High debt ratio: {debt_ratio:.1f} (should be <1.0)",
                    "Prioritize high-interest debt repayment"
                ))

            # Insurance recommendations
            insurance_adequacy = health_params['insurance_cover'] / (health_params['income'] * 10)
            if insurance_adequacy < 1.0:
                coverage_gap = (health_params['income'] * 10) - health_params['insurance_cover']
                recommendations.append((
                    "Increase Insurance",
                    f"₹{coverage_gap:,.0f} additional coverage needed",
                    "Consider term insurance for life coverage"
                ))

            # Expense recommendations
            expense_ratio = health_params['expenses'] / health_params['income']
            if expense_ratio > 0.6:
                recommendations.append((
                    "Reduce Expenses",
                    f"High expense ratio: {expense_ratio:.1f}% (should be <60%)",
                    "Track spending and identify areas to cut back"
                ))

            if not recommendations:
                st.markdown("""
                <div class="card health-card">
                    <h4>🎉 Excellent Financial Health!</h4>
                    <p>Your financial metrics are all in the optimal ranges. Keep up the good work!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for title, description, action in recommendations:
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <h4>{title}</h4>
                        <p>{description}</p>
                        <p><small>Action: {action}</small></p>
                    </div>
                    """, unsafe_allow_html=True)


class HistoricalAnalysis:
    @staticmethod
    def show_historical_analysis():
        st.header("📈 Historical Progress Analysis")
        user_id = get_user_id()

        # Get historical data
        historical_data = get_historical_data(user_id, months=12)

        if historical_data['financial'].empty:
            st.warning("No historical data available yet. Check back after using the app for a while.")
            return

        # Financial Metrics Over Time
        st.subheader("Financial Metrics Over Time")
        health_df = historical_data['health']
        if not health_df.empty:
            health_df['month'] = health_df['month'].astype(int)
            health_df['year'] = health_df['year'].astype(int)
            try:
                health_df['month_year'] = health_df.apply(
                    lambda x: f"{calendar.month_abbr[int(x['month'])]} {int(x['year'])}",
                    axis=1
                )
            except (ValueError, KeyError) as e:
                st.error(f"Error processing dates: {str(e)}")

        # Prepare financial data
        financial_df = historical_data['financial']
        if not financial_df.empty:
            financial_df['month'] = financial_df['month'].astype(int)
            financial_df['year'] = financial_df['year'].astype(int)
            financial_df['month_year'] = financial_df.apply(
                lambda x: f"{calendar.month_abbr[x['month']]} {x['year']}",
                axis=1
            )
            # Calculate ratios
            financial_df['savings_rate'] = financial_df['savings'] / financial_df['income'] * 100
            financial_df['expense_ratio'] = financial_df['expenses'] / financial_df['income'] * 100
            financial_df['investment_ratio'] = financial_df['investments'] / financial_df['income'] * 100

            # Plot metrics
            fig = px.line(financial_df,
                          x='month_year',
                          y=['savings_rate', 'expense_ratio', 'investment_ratio'],
                          labels={'value': 'Percentage (%)', 'month_year': 'Month'},
                          title="Key Financial Ratios Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Spending Pattern Changes
        st.subheader("Spending Pattern Evolution")

        spending_df = historical_data['spending']
        if not spending_df.empty:
            spending_df['month'] = spending_df['month'].astype(int)
            spending_df['year'] = spending_df['year'].astype(int)
            spending_df['month_year'] = spending_df.apply(
                lambda x: f"{calendar.month_abbr[x['month']]} {x['year']}",
                axis=1
            )

            # Get top 5 categories by average spending
            top_categories = spending_df.groupby('category')['amount'].mean().nlargest(5).index.tolist()
            filtered_spending = spending_df[spending_df['category'].isin(top_categories)]

            monthly_spending = filtered_spending.groupby(['month_year', 'category'])['amount'].mean().unstack()

            fig = px.line(monthly_spending,
                          labels={'value': 'Amount (₹)', 'month_year': 'Month'},
                          title="Monthly Spending by Top Categories")
            st.plotly_chart(fig, use_container_width=True)

        # Optimization Impact Analysis
        optimizations_df = historical_data['optimizations']
        if not optimizations_df.empty:
            optimizations_df['month'] = optimizations_df['month'].astype(int)
            optimizations_df['year'] = optimizations_df['year'].astype(int)
            optimizations_df['month_year'] = optimizations_df.apply(
                lambda x: f"{calendar.month_abbr[x['month']]} {x['year']}",
                axis=1
            )

            # Calculate savings from optimizations
            optimizations_df['savings'] = optimizations_df['current_amount'] - optimizations_df['optimized_amount']

            # Aggregate by month
            monthly_savings = optimizations_df.groupby('month_year').agg({
                'savings': 'sum'
            }).reset_index()

            # Calculate cumulative savings
            monthly_savings['cumulative_savings'] = monthly_savings['savings'].cumsum()

            fig = px.area(monthly_savings,
                          x='month_year',
                          y='cumulative_savings',
                          labels={'cumulative_savings': 'Savings (₹)', 'month_year': 'Month'},
                          title="Total Savings from Optimizations")
            st.plotly_chart(fig, use_container_width=True)

        # Financial Health Progress
        st.subheader("Financial Health Score Trend")

        health_df = historical_data['health']
        if not health_df.empty:
            health_df['month_year'] = health_df.apply(lambda x: f"{calendar.month_abbr[x['month']]} {x['year']}",
                                                      axis=1)

            fig = px.line(
                health_df,
                x='month_year',
                y='total_score',
                labels={'total_score': 'Health Score', 'month_year': 'Month'},
                title="Financial Health Score Over Time",
                markers=True)
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent")
            fig.add_hline(y=60, line_dash="dash", line_color="blue", annotation_text="Good")
            fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Fair")
            st.plotly_chart(fig, use_container_width=True)

            # Component scores radar chart
            latest_scores = health_df.iloc[-1]
            first_scores = health_df.iloc[0]

            categories = ['Savings', 'Investments', 'Debt', 'Insurance', 'Expenses']

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=[
                    latest_scores['savings_score'],
                    latest_scores['investment_score'],
                    latest_scores['debt_score'],
                    latest_scores['insurance_score'],
                    latest_scores['expense_score']
                ],
                theta=categories,
                fill='toself',
                name='Current'
            ))

            fig.add_trace(go.Scatterpolar(
                r=[
                    first_scores['savings_score'],
                    first_scores['investment_score'],
                    first_scores['debt_score'],
                    first_scores['insurance_score'],
                    first_scores['expense_score']
                ],
                theta=categories,
                fill='toself',
                name=f'{first_scores["month_year"]}'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 25]  # Max score for any component
                    )),
                showlegend=True,
                title="Financial Health Components: Then vs Now"
            )

            st.plotly_chart(fig, use_container_width=True)


class Leaderboard:
    @staticmethod
    def show_leaderboard(user_id):
        st.header("🏆 Financial Health Leaderboard")

        # Get current month/year
        current_month = datetime.now().month
        current_year = datetime.now().year

        # Get user rank and leaderboard
        user_rank, total_users, user_score = get_user_rank(user_id)
        leaderboard = get_leaderboard(10)

        # Show user's position
        if user_rank:
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%); color: white;">
                <h3>Your Position: #{user_rank} of {total_users}</h3>
                <p>Your Score: {user_score:.0f}/100</p>
                {f"🎉 Top 10%!" if user_rank / total_users <= 0.1 else ""}
                {f"🏅 Top Performer!" if user_rank == 1 else ""}
            </div>
            """, unsafe_allow_html=True)

            # Reward for top performer
            if user_rank == 1:
                st.balloons()
                st.success("""
                🎊 Congratulations! You're the top performer this month!

                **Your Reward:** Free financial consultation session with our expert
                """)

        # Show leaderboard table
        st.subheader("Top Performers")

        if not leaderboard.empty:
            # Format email to hide sensitive info
            leaderboard['email'] = leaderboard['email'].apply(
                lambda x: x[0] + "****" + x[x.find("@"):]
            )

            # Add medal emojis - fixed implementation
            ranks = []
            for i in range(1, len(leaderboard) + 1):
                if i == 1:
                    ranks.append("🥇")
                elif i == 2:
                    ranks.append("🥈")
                elif i == 3:
                    ranks.append("🥉")
                else:
                    ranks.append(f"{i}.")

            leaderboard['Rank'] = ranks

            # Reorder columns
            leaderboard = leaderboard[['Rank', 'email', 'score']]

            # Display as table with styling
            st.dataframe(
                leaderboard,
                column_config={
                    "Rank": "Rank",
                    "email": "User",
                    "score": st.column_config.NumberColumn(
                        "Score",
                        format="%.0f",
                        help="Financial Health Score (0-100)"
                    )
                },
                hide_index=True,
                use_container_width=True
            )

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Score Distribution")
                fig = px.histogram(
                    leaderboard,
                    x='score',
                    nbins=10,
                    labels={'score': 'Health Score'},
                    color_discrete_sequence=['#2563EB']
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Your Progress")
                history = pd.read_sql('''
                    SELECT month, year, score 
                    FROM leaderboard 
                    WHERE user_id = ?
                    ORDER BY year, month
                ''', init_db(), params=(user_id,))

                if not history.empty:
                    # Convert month to integer and handle any potential NaN values
                    history['month'] = history['month'].astype(float).astype(int)
                    history['year'] = history['year'].astype(int)

                    # Create date string safely
                    def format_date(row):
                        try:
                            return f"{calendar.month_abbr[row['month']]} {row['year']}"
                        except (ValueError, TypeError):
                            return "Unknown"

                    history['date'] = history.apply(format_date, axis=1)

                    fig = px.line(
                        history,
                        x='date',
                        y='score',
                        markers=True,
                        labels={'score': 'Health Score', 'date': 'Month'},
                        title="Your Score Over Time"
                    )
                    fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="green", opacity=0.1)
                    fig.add_hrect(y0=60, y1=80, line_width=0, fillcolor="blue", opacity=0.1)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No leaderboard data available yet")


class BusinessToolsUI:
    @staticmethod
    def show_business_tools():
        st.header("🚀 Business Optimization Tools")
        user_id = get_user_id()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "GST Advisor", "Invoice Discounting",
            "Working Capital", "Customer LTV",
            "Vendor Management"
        ])

        with tab1:
            st.subheader("🛍️ GST Optimization Advisor")
            form = st.form("gst_form")
            with form:
                turnover = st.number_input("Annual Turnover (₹)", min_value=0, value=5000000, step=100000)

                # Initialize with one row
                if 'gst_expenses' not in st.session_state:
                    st.session_state.gst_expenses = [
                        {"description": "Office Supplies", "amount": 50000, "gst_eligible": True}]

                # Display editable table
                for i, expense in enumerate(st.session_state.gst_expenses):
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.session_state.gst_expenses[i]["description"] = st.text_input(
                            "Description",
                            value=expense["description"],
                            key=f"gst_desc_{i}"
                        )
                    with cols[1]:
                        st.session_state.gst_expenses[i]["amount"] = st.number_input(
                            "Amount (₹)",
                            min_value=0,
                            value=expense["amount"],
                            key=f"gst_amt_{i}"
                        )
                    with cols[2]:
                        st.session_state.gst_expenses[i]["gst_eligible"] = st.checkbox(
                            "GST Eligible",
                            value=expense["gst_eligible"],
                            key=f"gst_elig_{i}"
                        )

                # Add/remove rows
                cols = st.columns(2)
                with cols[0]:
                    if st.form_submit_button("➕ Add Expense"):
                        st.session_state.gst_expenses.append({"description": "", "amount": 0, "gst_eligible": False})
                        st.rerun()
                with cols[1]:
                    if st.form_submit_button("➖ Remove Last"):
                        if len(st.session_state.gst_expenses) > 1:
                            st.session_state.gst_expenses.pop()
                            st.rerun()

                calculate_btn = st.form_submit_button("Calculate GST Liability")

                if calculate_btn:
                    result = BusinessTools.gst_advisor(turnover, st.session_state.gst_expenses)
                    col1, col2 = st.columns(2)
                    col1.metric("Monthly GST Liability", f"₹{result['liability'] / 12:,.0f}")
                    col2.metric("Input Tax Credit", f"₹{result['input_tax']:,.0f}")
                    if result['suggestion']:
                        st.success(result['suggestion'])

                    # Save to database
                    save_business_tool_data(
                        user_id,
                        "GST Advisor",
                        {"turnover": turnover, "expenses": st.session_state.gst_expenses},
                        result
                    )

        with tab2:
            st.subheader("🧾 Invoice Discounting Tool")
            with st.form("invoice_form"):
                amt = st.number_input("Invoice Amount (₹)", min_value=0, value=100000)
                days = st.slider("Days Until Due", 1, 90, 30)
                rate = st.slider("Discount Rate %", 1.0, 10.0, 2.0) / 100

                submitted = st.form_submit_button("Calculate")

                if submitted:
                    result = BusinessTools.invoice_discounting(amt, days, rate)
                    st.write(f"**Get ₹{result['discounted_amount']:,.0f} now** vs ₹{amt:,.0f} in {days} days")
                    st.metric("Annualized Return", f"{result['annualized_return']:.1f}%")

                    # Save to database
                    save_business_tool_data(
                        user_id,
                        "Invoice Discounting",
                        {"invoice_amount": amt, "days_remaining": days, "discount_rate": rate},
                        result
                    )

        with tab3:
            st.subheader("📦 Working Capital Optimizer")
            with st.form("wc_form"):
                col1, col2, col3 = st.columns(3)
                inventory = col1.number_input("Inventory Days", value=45)
                receivables = col2.number_input("Receivable Days", value=30)
                payables = col3.number_input("Payable Days", value=15)
                sales = st.number_input("Daily Sales (₹)", value=50000)

                submitted = st.form_submit_button("Analyze")

                if submitted:
                    result = BusinessTools.working_capital_analysis(inventory, receivables, payables, sales)
                    st.warning(
                        f"⚠️ Capital Blocked for {result['cycle_days']} Days (₹{result['required_capital']:,.0f})")
                    for suggestion in result['suggestions']:
                        st.info(f"💡 {suggestion}")

                    # Save to database
                    save_business_tool_data(
                        user_id,
                        "Working Capital",
                        {"inventory_days": inventory, "receivable_days": receivables,
                         "payable_days": payables, "daily_sales": sales},
                        result
                    )

        with tab4:
            st.subheader("👥 Customer Lifetime Value Calculator")
            with st.form("clv_form"):
                aov = st.number_input("Average Order Value (₹)", value=5000)
                freq = st.number_input("Purchases/Year", value=4)
                lifespan = st.number_input("Avg Customer Years", value=3.0)

                submitted = st.form_submit_button("Calculate")

                if submitted:
                    result = BusinessTools.calculate_clv(aov, freq, lifespan)
                    col1, col2 = st.columns(2)
                    col1.metric("Customer Lifetime Value", f"₹{result['clv']:,.0f}")
                    col2.metric("Max Recommended CAC", f"₹{result['max_cac']:,.0f}")

                    # Save to database
                    save_business_tool_data(
                        user_id,
                        "Customer LTV",
                        {"avg_order_value": aov, "purchase_freq": freq, "avg_customer_lifespan": lifespan},
                        result
                    )

        with tab5:
            st.subheader("🤝 Vendor Scorecard")
            with st.form("vendor_form"):
                col1, col2, col3, col4 = st.columns(4)
                price = col1.slider("Price", 0, 100, 60)
                quality = col2.slider("Quality", 0, 100, 80)
                delivery = col3.slider("On-Time Delivery", 0, 100, 90)
                terms = col4.slider("Payment Terms", 0, 100, 30)

                submitted = st.form_submit_button("Evaluate")

                if submitted:
                    result = BusinessTools.vendor_scorecard(price, quality, delivery, terms)
                    st.metric("Overall Score", f"{result['score']}/100 ({result['grade']})")
                    if result['improve']:
                        st.error(f"Focus Area: {result['improve']}")

                    # Save to database
                    save_business_tool_data(
                        user_id,
                        "Vendor Scorecard",
                        {"price": price, "quality": quality, "delivery": delivery, "payment_terms": terms},
                        result
                    )


# ======================
# Debt Management
# ======================
class DebtManager:
    @staticmethod
    def calculate_debt_snowball(user_id):
        conn = init_db()
        c = conn.cursor()

        c.execute('''SELECT id, debt_name, principal, interest_rate, monthly_payment, remaining_term
                     FROM debts
                     WHERE user_id = ? AND is_active = 1
                     ORDER BY principal ASC''', (user_id,))

        debts = c.fetchall()
        return debts

    @staticmethod
    def calculate_debt_avalanche(user_id):
        conn = init_db()
        c = conn.cursor()

        c.execute('''SELECT id, debt_name, principal, interest_rate, monthly_payment, remaining_term
                     FROM debts
                     WHERE user_id = ? AND is_active = 1
                     ORDER BY interest_rate DESC''', (user_id,))

        debts = c.fetchall()
        return debts

    @staticmethod
    def calculate_payoff_plan(user_id, method='snowball'):
        if method == 'snowball':
            debts = DebtManager.calculate_debt_snowball(user_id)
        else:
            debts = DebtManager.calculate_debt_avalanche(user_id)

        # Simplified payoff calculation
        payoff_plan = []
        extra_payment = 5000  # Example extra payment

        for debt in debts:
            id, name, principal, rate, monthly_payment, term = debt
            new_payment = monthly_payment + extra_payment

            # Calculate months to payoff
            months = 0
            balance = principal

            while balance > 0 and months < 120:  # Cap at 10 years
                interest = balance * (rate / 12 / 100)
                balance = balance + interest - new_payment
                months += 1

            payoff_plan.append({
                'name': name,
                'current_term': term,
                'projected_term': months,
                'interest_saved': (term - months) * monthly_payment * rate / 12 / 100
            })

        return payoff_plan


# ======================
# Goal Tracking
# ======================
class GoalTracker:
    @staticmethod
    def create_goal(user_id, goal_name, target_amount, target_date, priority=3):
        conn = init_db()
        c = conn.cursor()

        c.execute('''INSERT INTO financial_goals
                     (user_id, goal_name, target_amount, current_amount, target_date, priority)
                     VALUES (?, ?, ?, 0, ?, ?)''',
                  (user_id, goal_name, target_amount, target_date, priority))

        conn.commit()

    @staticmethod
    def update_goal_progress(user_id, goal_id, amount):
        conn = init_db()
        c = conn.cursor()

        c.execute('''UPDATE financial_goals
                     SET current_amount = current_amount + ?
                     WHERE id = ? AND user_id = ?''',
                  (amount, goal_id, user_id))

        conn.commit()

    @staticmethod
    def get_goals(user_id):
        conn = init_db()
        c = conn.cursor()

        c.execute('''SELECT id, goal_name, target_amount, current_amount, target_date, priority, achieved
                     FROM financial_goals
                     WHERE user_id = ?
                     ORDER BY priority, target_date''', (user_id,))

        goals = []
        today = datetime.now().date()

        for goal in c.fetchall():
            id, name, target, current, date, priority, achieved = goal
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
            progress = (current / target) * 100 if target > 0 else 0
            days_remaining = (target_date - today).days

            goals.append({
                'id': id,
                'name': name,
                'target': target,
                'current': current,
                'progress': progress,
                'target_date': date,
                'days_remaining': days_remaining,
                'priority': priority,
                'achieved': achieved
            })

        return goals


# ======================
# Transaction Monitoring
# ======================
class TransactionMonitor:
    @staticmethod
    def detect_anomalies(user_id):
        conn = init_db()

        # Get recent transactions
        transactions = pd.read_sql('''SELECT amount, category, timestamp 
                                     FROM transactions 
                                     WHERE user_id = ? 
                                     ORDER BY timestamp DESC LIMIT 100''',
                                   conn, params=(user_id,))

        if len(transactions) < 10:
            return []

        # Train isolation forest for anomaly detection
        clf = IsolationForest(contamination=0.1)
        amounts = transactions['amount'].values.reshape(-1, 1)
        clf.fit(amounts)

        preds = clf.predict(amounts)
        transactions['is_anomaly'] = preds == -1

        # Get suspicious transactions
        anomalies = transactions[transactions['is_anomaly']].to_dict('records')

        # Update database with fraud flags
        for txn in anomalies:
            conn.execute('''UPDATE transactions 
                           SET is_fraud = 1 
                           WHERE user_id = ? AND timestamp = ?''',
                         (user_id, txn['timestamp']))

        conn.commit()
        return anomalies


# ======================
# Enhanced UI Components
# ======================
class EnhancedDashboard:
    @staticmethod
    def show_ai_recommendations(user_id):
        st.header("🤖 AI Financial Advisor")

        if st.button("Generate Personalized Recommendations"):
            advisor = AIAdvisor()
            user_data = st.session_state.financial_data

            with st.spinner("Analyzing your financial profile..."):
                savings_rec = advisor.generate_recommendation(user_data, "savings optimization")
                investment_rec = advisor.generate_recommendation(user_data, "investment strategy")
                debt_rec = advisor.generate_recommendation(user_data, "debt management")

                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"""
                    <div class="card">
                        <h4>💸 Savings Recommendation</h4>
                        <p>{savings_rec}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with cols[1]:
                    st.markdown(f"""
                    <div class="card">
                        <h4>📈 Investment Recommendation</h4>
                        <p>{investment_rec}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with cols[2]:
                    st.markdown(f"""
                    <div class="card">
                        <h4>🏦 Debt Recommendation</h4>
                        <p>{debt_rec}</p>
                    </div>
                    """, unsafe_allow_html=True)

    @staticmethod
    def show_investment_dashboard(user_id):
        st.header("📊 Investment Portfolio")

        # Update portfolio values
        if st.button("🔄 Update Market Values"):
            with st.spinner("Fetching latest market data..."):
                InvestmentTracker.update_portfolio_value(user_id)

        portfolio = InvestmentTracker.get_portfolio_performance(user_id)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"₹{portfolio['total_value']:,.0f}")
        col2.metric("Total Invested", f"₹{portfolio['total_invested']:,.0f}")
        col3.metric("Overall Returns", f"{portfolio['returns']:.1f}%",
                    f"{'↑' if portfolio['returns'] >= 0 else '↓'} {abs(portfolio['returns']):.1f}%")

        # Portfolio allocation
        st.subheader("Asset Allocation")
        fig = px.pie(
            names=[x[0] for x in portfolio['portfolio']],
            values=[x[1] for x in portfolio['portfolio']],
            hole=0.4,
            title="Portfolio Composition"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add new investment
        with st.expander("➕ Add New Investment"):
            with st.form("new_investment"):
                asset_type = st.selectbox("Asset Type", ["Equity", "Mutual Fund", "ETF", "Bonds", "Gold"])
                asset_name = st.text_input("Ticker/Name", "RELIANCE.NS")
                quantity = st.number_input("Quantity", min_value=0.001, step=0.001, format="%.3f")
                buy_price = st.number_input("Buy Price (₹)", min_value=0.01)

                if st.form_submit_button("Add to Portfolio"):
                    conn = init_db()
                    c = conn.cursor()

                    current_price = InvestmentTracker.fetch_live_price(asset_name) or buy_price
                    current_value = current_price * quantity

                    c.execute('''INSERT INTO investment_portfolio
                                (user_id, asset_type, asset_name, quantity, buy_price, current_value)
                                VALUES (?, ?, ?, ?, ?, ?)''',
                              (user_id, asset_type, asset_name, quantity, buy_price, current_value))

                    conn.commit()
                    st.success("Investment added successfully!")
                    st.rerun()

    @staticmethod
    def show_debt_dashboard(user_id):
        st.header("🏦 Debt Management")

        tab1, tab2, tab3 = st.tabs(["Current Debts", "Snowball Method", "Avalanche Method"])

        with tab1:
            conn = init_db()
            debts = pd.read_sql('''SELECT debt_name, principal, interest_rate, monthly_payment, remaining_term 
                                  FROM debts 
                                  WHERE user_id = ? AND is_active = 1''',
                                conn, params=(user_id,))

            if debts.empty:
                st.info("No active debts found")
            else:
                st.dataframe(debts, hide_index=True)

                total_debt = debts['principal'].sum()
                monthly_payments = debts['monthly_payment'].sum()
                avg_interest = (debts['interest_rate'] * debts['principal']).sum() / total_debt

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Debt", f"₹{total_debt:,.0f}")
                col2.metric("Monthly Payments", f"₹{monthly_payments:,.0f}")
                col3.metric("Avg Interest Rate", f"{avg_interest:.2f}%")

                # Add new debt
                with st.expander("Add New Debt"):
                    with st.form("new_debt"):
                        name = st.text_input("Debt Name")
                        principal = st.number_input("Principal Amount (₹)", min_value=0.01)
                        rate = st.number_input("Interest Rate (%)", min_value=0.01, max_value=99.99, step=0.01)
                        payment = st.number_input("Monthly Payment (₹)", min_value=0.01)
                        term = st.number_input("Remaining Term (months)", min_value=1)
                        lender = st.text_input("Lender")

                        if st.form_submit_button("Add Debt"):
                            c = conn.cursor()
                            c.execute('''INSERT INTO debts
                                        (user_id, debt_name, principal, interest_rate, monthly_payment, remaining_term, lender)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                      (user_id, name, principal, rate, payment, term, lender))
                            conn.commit()
                            st.success("Debt added successfully!")
                            st.rerun()

        with tab2:
            st.markdown("""
            **❄️ Debt Snowball Method**  
            Pay off debts from smallest to largest balance, gaining momentum as each balance is paid off.
            """)

            snowball_plan = DebtManager.calculate_payoff_plan(user_id, 'snowball')
            if not snowball_plan:
                st.info("No debts to analyze")
            else:
                for debt in snowball_plan:
                    st.markdown(f"""
                    <div class="card">
                        <h4>{debt['name']}</h4>
                        <p>Current payoff term: {debt['current_term']} months</p>
                        <p>Projected payoff term: {debt['projected_term']} months</p>
                        <p>Interest saved: ₹{debt['interest_saved']:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            st.markdown("""
            **🏔️ Debt Avalanche Method**  
            Pay off debts with the highest interest rates first to save money on interest payments.
            """)

            avalanche_plan = DebtManager.calculate_payoff_plan(user_id, 'avalanche')
            if not avalanche_plan:
                st.info("No debts to analyze")
            else:
                for debt in avalanche_plan:
                    st.markdown(f"""
                    <div class="card">
                        <h4>{debt['name']}</h4>
                        <p>Current payoff term: {debt['current_term']} months</p>
                        <p>Projected payoff term: {debt['projected_term']} months</p>
                        <p>Interest saved: ₹{debt['interest_saved']:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)

    @staticmethod
    def show_goal_tracker(user_id):
        st.header("🎯 Financial Goals")

        goals = GoalTracker.get_goals(user_id)

        if not goals:
            st.info("No financial goals set yet")
        else:
            for goal in goals:
                st.markdown(f"""
                <div class="card">
                    <h3>{goal['name']}</h3>
                    <p>Target: ₹{goal['target']:,.0f} by {goal['target_date']}</p>
                    <p>Progress: ₹{goal['current']:,.0f} of ₹{goal['target']:,.0f} ({goal['progress']:.1f}%)</p>
                    <div class="goal-progress">
                        <div class="goal-progress-fill" style="width: {min(goal['progress'], 100)}%"></div>
                    </div>
                    <p>{goal['days_remaining']} days remaining</p>
                </div>
                """, unsafe_allow_html=True)

        # Add new goal
        with st.expander("➕ Set New Goal"):
            with st.form("new_goal"):
                name = st.text_input("Goal Name")
                target = st.number_input("Target Amount (₹)", min_value=0.01)
                date = st.date_input("Target Date", min_value=datetime.now().date())
                priority = st.select_slider("Priority", options=[1, 2, 3, 4, 5], value=3)

                if st.form_submit_button("Set Goal"):
                    GoalTracker.create_goal(user_id, name, target, date, priority)
                    st.success("Goal created successfully!")
                    st.rerun()

        # Update goal progress
        if goals:
            with st.expander("🔄 Update Progress"):
                with st.form("update_goal"):
                    goal_id = st.selectbox("Select Goal", [g['id'] for g in goals],
                                           format_func=lambda x: next(g['name'] for g in goals if g['id'] == x))
                    amount = st.number_input("Amount to Add (₹)", min_value=0.01)

                    if st.form_submit_button("Update"):
                        GoalTracker.update_goal_progress(user_id, goal_id, amount)
                        st.success("Progress updated!")
                        st.rerun()

    @staticmethod
    def show_transaction_monitor(user_id):
        st.header("💳 Transaction Monitoring")

        # Detect anomalies
        if st.button("🔍 Check for Suspicious Activity"):
            anomalies = TransactionMonitor.detect_anomalies(user_id)

            if not anomalies:
                st.success("No suspicious transactions found")
            else:
                st.warning(f"Found {len(anomalies)} potentially suspicious transactions")

                for txn in anomalies:
                    st.markdown(f"""
                    <div class="card fraud-alert">
                        <h4>⚠️ Suspicious Transaction</h4>
                        <p>Amount: ₹{txn['amount']:,.0f}</p>
                        <p>Category: {txn['category']}</p>
                        <p>Date: {txn['timestamp']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Recent transactions
        st.subheader("Recent Transactions")
        conn = init_db()
        transactions = pd.read_sql('''SELECT description, amount, category, timestamp, is_fraud 
                                     FROM transactions 
                                     WHERE user_id = ?
                                     ORDER BY timestamp DESC LIMIT 20''',
                                   conn, params=(user_id,))

        if transactions.empty:
            st.info("No transactions recorded yet")
        else:
            # Format fraud alerts
            transactions['alert'] = transactions['is_fraud'].apply(lambda x: "⚠️" if x else "")

            # Show data table
            st.dataframe(
                transactions[['alert', 'description', 'amount', 'category', 'timestamp']],
                column_config={
                    "alert": "Alert",
                    "description": "Description",
                    "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                    "category": "Category",
                    "timestamp": "Date"
                },
                hide_index=True
            )

        # Add manual transaction
        with st.expander("➕ Add Manual Transaction"):
            with st.form("add_transaction"):
                desc = st.text_input("Description")
                amount = st.number_input("Amount (₹)", min_value=0.01)
                category = st.selectbox("Category", ["Food", "Shopping", "Transport", "Entertainment", "Utilities"])
                date = st.date_input("Date", value=datetime.now().date())

                if st.form_submit_button("Add Transaction"):
                    c = conn.cursor()
                    c.execute('''INSERT INTO transactions
                                (user_id, description, amount, category, timestamp)
                                VALUES (?, ?, ?, ?, ?)''',
                              (user_id, desc, amount, category, f"{date} 12:00:00"))
                    conn.commit()
                    st.success("Transaction added successfully!")
                    st.rerun()


# ======================
# Main Application
# ======================
def init_session_state():
    """Initialize session state variables"""
    if 'financial_data' not in st.session_state:
        # Try to load current month data
        user_id = get_user_id()
        if user_id:
            current_data = get_current_month_data(user_id)

            if current_data['financial'] is not None:
                # Load existing data
                st.session_state.financial_data = current_data['financial']
                st.session_state.spending_data = current_data['spending']
            else:
                # Initialize with default values
                st.session_state.financial_data = {
                    'income': 0,
                    'expenses': 0,
                    'savings': 0,
                    'investments': 0,
                    'liabilities': 0,
                    'insurance_cover': 0,
                    'insurance_premium': 0,
                    'tax_regime': 'new',
                    'age': 30,
                    'dependents': 0
                }
                st.session_state.spending_data = {}

    if 'optimized_budget' not in st.session_state:
        st.session_state.optimized_budget = None

    if 'min_savings' not in st.session_state:
        st.session_state.min_savings = 0

    if 'risk_appetite' not in st.session_state:
        st.session_state.risk_appetite = 0.5

    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False


def spending_input_form():
    """Interactive spending input form with better UX"""
    with st.expander("💰 Edit Spending Categories", expanded=True):
        cols = st.columns([2, 1, 1.2, 1, 1])
        cols[0].subheader("Category")
        cols[1].subheader("Amount (₹)")
        cols[2].subheader("Happiness (1-10)")
        cols[3].subheader("Essential")
        cols[4].subheader("Actions")

        to_delete = []

        for i, cat in enumerate(list(st.session_state.spending_data.keys())):
            cols = st.columns([2, 1, 1, 1, 1])
            new_name = cols[0].text_input(
                f"cat_name_{i}",
                value=cat,
                label_visibility="collapsed",
                key=f"cat_name_{i}"
            )

            new_amount = cols[1].number_input(
                f"amt_{i}",
                min_value=0,
                value=int(st.session_state.spending_data[cat]['amount']),
                step=500,
                label_visibility="collapsed",
                key=f"amt_{i}"
            )

            new_happy = cols[2].slider(
                f"happy_{i}",
                1, 10,
                st.session_state.spending_data[cat]['happiness'],
                label_visibility="collapsed",
                key=f"happy_{i}"
            )

            essential = cols[3].checkbox(
                "",
                value=st.session_state.spending_data[cat].get('essential', False),
                key=f"ess_{i}"
            )

            if cols[4].button("❌", key=f"del_{i}"):
                to_delete.append(cat)

            # Update values
            if new_name != cat:
                st.session_state.spending_data[new_name] = st.session_state.spending_data.pop(cat)

            st.session_state.spending_data[new_name].update({
                'amount': new_amount,
                'happiness': new_happy,
                'essential': essential
            })

        # Handle deletions
        for cat in to_delete:
            del st.session_state.spending_data[cat]

        # Add new category
        new_cols = st.columns([2, 2, 2, 1, 1])
        new_cat = new_cols[0].text_input(
            "Add new category",
            value="",
            placeholder="Category name",
            label_visibility="collapsed",
            key="new_cat"
        )

        if new_cat and new_cat not in st.session_state.spending_data:
            new_cols[1].number_input(
                "Amount",
                min_value=0,
                value=3000,
                step=500,
                label_visibility="collapsed",
                key="new_amt"
            )
            new_cols[2].slider(
                "Happiness",
                1, 10, 5,
                label_visibility="collapsed",
                key="new_happy"
            )
            new_cols[3].checkbox(
                "Essential",
                value=False,
                key="new_ess"
            )

            if new_cols[4].button("➕ Add"):
                st.session_state.spending_data[new_cat] = {
                    'amount': st.session_state.new_amt,
                    'happiness': st.session_state.new_happy,
                    'essential': st.session_state.new_ess
                }
                st.rerun()


def show_personal_finance(user_id):
    st.title("💰 OptiSpend Pro+")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2563EB 0%, #059669 100%); padding: 1rem; border-radius: 10px; color: white;">
    <h3 style="color: white; margin: 0;">AI-Powered Financial Optimization Suite</h3>
    <p style="margin: 0.5rem 0 0;">Maximize happiness per rupee • Minimize wasteful spending • Optimize your financial future</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with user profile
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        st.session_state.financial_data['income'] = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            value=int(st.session_state.financial_data.get('income', 1)),
            step=1000,
            key="sidebar_income"
        )

        st.session_state.min_savings = st.number_input(
            "Monthly Savings Goal (₹)",
            min_value=0,
            value=st.session_state.min_savings,
            step=1000,
            key="savings"
        )

        st.session_state.risk_appetite = st.slider(
            "Risk Appetite",
            0.0, 1.0, st.session_state.risk_appetite,
            help="Higher values prioritize happiness over savings stability",
            key="risk_appetite_sidebar"
        )
        # Save button
        if st.button("💾 Save Data", type="primary", key="main_save_button"):
            save_financial_data(user_id, st.session_state.financial_data)
            save_spending_data(user_id, st.session_state.spending_data)
            st.success("Data saved successfully!")

        # Dark mode toggle
        # st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        # if st.session_state.dark_mode:
        #     st.markdown("<style>[data-theme='dark']</style>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📊 Financial Profile")
        st.session_state.financial_data['tax_regime'] = st.radio(
            "Tax Regime",
            ['new', 'old'],
            index=0 if st.session_state.financial_data['tax_regime'] == 'new' else 1,
            format_func=lambda x: "New Regime" if x == 'new' else "Old Regime",
            key="tax_regime"
        )

        detailed_expander = st.expander("Detailed Profile")
        with detailed_expander:
            st.session_state.financial_data['age'] = st.number_input(
                "Your Age",
                min_value=18,
                max_value=70,
                value=int(st.session_state.financial_data.get('age', 30)),
                key="profile_age"
            )

            st.session_state.financial_data['dependents'] = st.number_input(
                "Number of Dependents",
                min_value=0,
                value=int(st.session_state.financial_data.get('dependents', 0)),
                key="profile_dependents"
            )

            st.session_state.financial_data['expenses'] = st.number_input(
                "Annual Expenses (₹)",
                value=int(st.session_state.financial_data.get('expenses', 0)),
                step=10000,
                key="profile_expenses"
            )

            st.session_state.financial_data['savings'] = st.number_input(
                "Annual Savings (₹)",
                value=int(st.session_state.financial_data.get('savings', 0)),
                step=10000,
                key="profile_savings"
            )

            st.session_state.financial_data['investments'] = st.number_input(
                "Total Investments (₹)",
                value=int(st.session_state.financial_data.get('investments', 0)),
                step=10000,
                key="profile_investments"
            )

            st.session_state.financial_data['liabilities'] = st.number_input(
                "Total Liabilities (₹)",
                value=int(st.session_state.financial_data.get('liabilities', 0)),
                step=100000,
                key="profile_liabilities"
            )

            st.session_state.financial_data['insurance_cover'] = st.number_input(
                "Total Insurance Cover (₹)",
                value=int(st.session_state.financial_data.get('insurance_cover', 0)),
                step=100000,
                key="profile_insurance_cover"
            )

            st.session_state.financial_data['insurance_premium'] = st.number_input(
                "Total Insurance Premium (₹)",
                value=int(st.session_state.financial_data.get('insurance_premium', 0)),
                step=1000,
                key="profile_insurance_premium"
            )

        st.markdown("---")
        st.markdown("## 💰 Spending Input")
        spending_input_form()

        # Save button
        if st.button("💾 Save Data", type="primary", key="save_data_main"):
            save_financial_data(user_id, st.session_state.financial_data)
            save_spending_data(user_id, st.session_state.spending_data)
            st.success("Data saved successfully!")

        # Report generation
        st.markdown("---")
        st.markdown("## 📄 Generate Report")
        ReportGenerator.show_sidebar_report_options(user_id)


    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 Dashboard",
        "💡 Optimization",
        "🔮 Forecast",
        "🧠 AI Advisor",
        "🧾 Tax Center",
        "🛡️ Insurance",
        "📈 Investments",
        "🏆 Financial Health",
        "🕰️ Historical Analysis",
        "🚀 Advanced Tools",
        "🏅 Leaderboard"
    ])

    with tab1:
        FinancialDashboard.show_main_dashboard()

    with tab2:
        BudgetOptimizer.show_optimization()

    with tab3:
        ForecastPlanner.show_forecast()

    with tab4:
        AIAdvisorUI.show_advisor()

    with tab5:
        TaxCenter.show_tax_center()

    with tab6:
        InsurancePlanner.show_insurance_analysis()

    with tab7:
        InvestmentPlanner.show_investment_analysis()

    with tab8:
        FinancialHealthAnalyzer.show_health_analysis()

    with tab9:
        HistoricalAnalysis.show_historical_analysis()

    with tab10:
        EnhancedDashboard.show_ai_recommendations(user_id)
        EnhancedDashboard.show_investment_dashboard(user_id)
        EnhancedDashboard.show_debt_dashboard(user_id)
        EnhancedDashboard.show_goal_tracker(user_id)
        EnhancedDashboard.show_transaction_monitor(user_id)

    with tab11:
        Leaderboard.show_leaderboard(user_id)


def show_login():
    st.title("🔒 OptiSpend PRO+ Login")

    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["✨ Register", "Login"])

    with tab1:
        with st.form("register_form"):
            st.subheader("Create New Account")
            email = st.text_input("Email Address")
            password = st.text_input("Create Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")

            # Add basic form validation
            if st.form_submit_button("Register"):
                if not email or not password or not confirm:
                    st.error("Please fill all fields")
                elif password != confirm:
                    st.error("Passwords don't match")
                else:
                    conn = init_db()
                    c = conn.cursor()

                    # Check if email exists
                    c.execute("SELECT 1 FROM users WHERE email = ?", (email,))
                    if c.fetchone():
                        st.error("Email already registered")
                    else:
                        salt = generate_salt()
                        pwd_hash = hash_password(password, salt)
                        user_id = f"user_{int(datetime.now().timestamp())}"

                        c.execute('''INSERT INTO users
                                    (user_id, email, password_hash, salt)
                                    VALUES (?, ?, ?, ?)''',
                                  (user_id, email, pwd_hash, salt))

                        conn.commit()
                        st.success("Registration successful! Please login")
                        st.session_state.login_active_tab = 1
                        st.rerun()

    with tab2:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Login"):
                conn = init_db()
                c = conn.cursor()

                c.execute("SELECT user_id, password_hash, salt FROM users WHERE email = ?", (email,))
                user = c.fetchone()

                if user and verify_password(user[1], password, user[2]):
                    st.session_state.authenticated = True
                    st.session_state.user_id = user[0]
                    init_session_state()  # Initialize session state after login
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    # Set default tab if not set
    if 'login_active_tab' not in st.session_state:
        st.session_state.login_active_tab = 0


def show_main_app(user_id):
    # Initialize session state
    init_session_state()

    # Navigation
    menu = ["Personal Finance", "Business Tools", "Reports"]
    choice = st.sidebar.selectbox("Dashboard", menu)

    # User Profile
    with st.sidebar:
        st.header("User Profile")
    # Main app sections
    if choice == "Personal Finance":
        show_personal_finance(user_id)
    elif choice == "Business Tools":
        BusinessToolsUI.show_business_tools()
    elif choice == "Reports":
        current_month = datetime.now().month
        current_year = datetime.now().year
        report = ReportGenerator.generate_monthly_report(user_id, current_month, current_year)
        if report:
            st.download_button(
                label="📊 Download Monthly Report",
                data=report,
                file_name=f"OptiSpend_Report_{current_month}_{current_year}.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Failed to generate report")



def main():
    # Initialize database
    init_db()

    # Check authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_login()
    else:
        user_id = st.session_state.user_id
        show_main_app(user_id)


if __name__ == "__main__":
    main()