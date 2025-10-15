from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import sqlite3
import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv
import ssl
import urllib3
from datetime import datetime

# Load environment variables
load_dotenv()

# Disable SSL warnings for development (if needed)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Simple message classes to replace LangChain messages
class SystemMessage:
    def __init__(self, content):
        self.content = content
        self.type = "system"

class HumanMessage:
    def __init__(self, content):
        self.content = content
        self.type = "human"

# Configuration class for DeepSeek
class Config:
    # DeepSeek Configuration from environment variables
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

class DeepSeekClient:
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.base_url = Config.DEEPSEEK_BASE_URL
        self.model = Config.DEEPSEEK_MODEL
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    def invoke(self, messages, timeout=120, max_tokens=4000, retry_count=2):
        """
        Invoke DeepSeek API with messages, with retry logic and configurable timeout
        """
        # Convert messages to DeepSeek format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Handle LangChain message objects
                if hasattr(msg, 'type'):
                    role = "system" if msg.type == "system" else "user"
                else:
                    role = "user"
                formatted_messages.append({
                    "role": role,
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0,
            "max_tokens": max_tokens
        }
        
        last_error = None
        
        # Retry logic for handling timeouts and temporary errors
        for attempt in range(retry_count + 1):
            try:
                print(f"🔄 DeepSeek API call attempt {attempt + 1}/{retry_count + 1} (timeout: {timeout}s)")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    verify=False  # For development - set to True for production
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Return object with content attribute to maintain compatibility
                class DeepSeekResponse:
                    def __init__(self, content):
                        self.content = content
                
                print(f"✅ DeepSeek API call successful on attempt {attempt + 1}")
                return DeepSeekResponse(result['choices'][0]['message']['content'])
                
            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"⏰ DeepSeek API timeout on attempt {attempt + 1}: {e}")
                if attempt < retry_count:
                    print(f"🔄 Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                    # Increase timeout for next attempt
                    timeout = min(timeout + 30, 180)
                continue
                
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"❌ DeepSeek API Error on attempt {attempt + 1}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response text: {e.response.text}")
                if attempt < retry_count:
                    print(f"🔄 Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                    continue
                break
        
        # If all retries failed, raise the last error
        print(f"❌ All DeepSeek API attempts failed")
        raise last_error

# Define the state structure for LangGraph
class SQLAgentState(TypedDict):
    user_question: str
    conversation_history: List[dict]
    table_schema: str
    sql_query: str
    query_result: Optional[pd.DataFrame]
    error_message: Optional[str]
    attempt_count: int
    max_attempts: int
    final_answer: str
    requires_clarification: bool
    clarification_question: str
    visualization_data: Optional[dict]
    messages: List[dict]
    result_type: Optional[str]  # 'data', 'visualization', 'text'
    previous_code_errors: List[str]  # Track visualization code execution errors

class SQLAgent:
    def __init__(self, max_attempts: int = 5, db_path: str = None):
        """
        Initialize SQL Agent with LangGraph workflow using Azure OpenAI
        
        Args:
            max_attempts: Maximum query refinement attempts
            db_path: Path to existing SQLite database file (defaults to ../data/database.db)
        """
        # Set default database path relative to script location
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(script_dir, "data", "database.db")
        else:
            self.db_path = db_path
        self.max_attempts = max_attempts
        self.connection = None
        
        # Initialize DeepSeek client
        self.client = DeepSeekClient()
        print(f"✅ DeepSeek client initialized with model: {Config.DEEPSEEK_MODEL}")
        
        # Load data and get schema
        self._initialize_database()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _initialize_database(self):
        """Connect to existing database and get schema"""
        print("🔄 Connecting to existing database...")
        print(f"📍 Database path: {self.db_path}")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        # Verify database connection works but don't store it
        with sqlite3.connect(self.db_path) as test_conn:
            # Just a quick test query to ensure connection works
            test_conn.execute("SELECT 1")
        
        # Get the single table name and schema
        tables = self._list_tables()
        if not tables:
            raise ValueError("No tables found in database")
        
        # Use the first (and should be only) table
        self.table_name = tables[0]
        self.table_columns = self._describe_table(self.table_name)
        
        if self.table_columns is None:
            raise ValueError(f"Could not read schema for table: {self.table_name}")
        
        print(f"✅ Connected to database with table: {self.table_name}")
        print(f"📊 Table has {len(self.table_columns)} columns")
        
        # Debug: Print column names to help identify the exact column names
        print("� Available columns:")
        for _, col in self.table_columns.iterrows():
            print(f"  - '{col['name']}' ({col['type']})")

    
    def _list_tables(self):
        """List all tables in the database"""
        # Create a new connection to avoid thread issues
        with sqlite3.connect(self.db_path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", 
                conn
            )
        return tables['name'].tolist()
    
    def _describe_table(self, table_name):
        """Get column information for a table"""
        try:
            # Create a new connection to avoid thread issues
            with sqlite3.connect(self.db_path) as conn:
                columns = pd.read_sql_query(f"PRAGMA table_info(`{table_name}`)", conn)
            return columns[['name', 'type']]
        except Exception as e:
            print(f"❌ Error describing table {table_name}: {str(e)}")
            return None
    
    def _query_database(self, sql_query):
        """Execute SQL query and return results"""
        try:
            # Create a new connection for this specific query to avoid thread issues
            with sqlite3.connect(self.db_path) as conn:
                result = pd.read_sql_query(sql_query, conn)
            return result
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return None
    
    def _get_schema_string(self) -> str:
        """Get database schema as formatted string for single table"""
        schema_parts = [
            f"Table: {self.table_name}",
            "Columns:"
        ]
        
        for _, col in self.table_columns.iterrows():
            schema_parts.append(f"  - {col['name']} ({col['type']})")
        
        return "\n".join(schema_parts)
    
    def _get_dynamic_table_metadata(self, query_specific_columns=None):
        """
        Dynamically fetch table metadata including unique values and date formats
        
        Args:
            query_specific_columns: List of column names to prioritize for detailed analysis
        """
        print("🔄 Fetching comprehensive dynamic table metadata...")
        
        metadata = {
            'columns': [],
            'categorical_values': {},
            'date_formats': {},
            'numeric_stats': {},
            'sample_data': None,
            'row_count': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get row count
                row_count_query = f"SELECT COUNT(*) as count FROM `{self.table_name}`"
                metadata['row_count'] = pd.read_sql_query(row_count_query, conn).iloc[0]['count']
                
                # Get column information
                columns_query = f"PRAGMA table_info(`{self.table_name}`)"
                columns_df = pd.read_sql_query(columns_query, conn)
                
                # Get sample data for analysis
                sample_query = f"SELECT * FROM `{self.table_name}` LIMIT 3"
                metadata['sample_data'] = pd.read_sql_query(sample_query, conn)
                
                # Priority columns - analyze these in detail first
                priority_columns = []
                if query_specific_columns:
                    priority_columns.extend(query_specific_columns)
                
                # Always include key analytical columns
                key_columns = ['Complaint Type', 'Status', 'Agency', 'Borough', 'Created Date', 'Closed Date', 
                              'Descriptor', 'Location Type', 'City', 'Agency Name']
                priority_columns.extend([col for col in key_columns if col not in priority_columns])
                
                # Analyze each column
                analyzed_count = 0
                for _, col in columns_df.iterrows():
                    col_name = col['name']
                    col_type = col['type']
                    
                    metadata['columns'].append({
                        'name': col_name,
                        'type': col_type,
                        'sql_type': col_type
                    })
                    
                    # Determine if this column should get detailed analysis
                    is_priority = col_name in priority_columns
                    is_key_analytical = any(key in col_name.lower() for key in 
                                          ['complaint', 'type', 'status', 'agency', 'date', 'borough', 'city'])
                    
                    # Always analyze priority columns and key analytical columns
                    # For others, use smart sampling based on column type and name patterns
                    should_analyze = is_priority or is_key_analytical or analyzed_count < 20
                    
                    if should_analyze:
                        try:
                            # For TEXT columns, determine if date or categorical
                            if col_type.upper() == 'TEXT':
                                # Check if it's a date column
                                if any(date_word in col_name.lower() for date_word in 
                                      ['date', 'time', 'created', 'closed', 'updated', 'due']):
                                    # Sample date formats from this column
                                    date_sample_query = f"""
                                    SELECT DISTINCT `{col_name}` as sample_date 
                                    FROM `{self.table_name}` 
                                    WHERE `{col_name}` IS NOT NULL 
                                    ORDER BY `{col_name}` DESC
                                    LIMIT 8
                                    """
                                    date_samples = pd.read_sql_query(date_sample_query, conn)
                                    if not date_samples.empty:
                                        formats = []
                                        null_count_query = f"SELECT COUNT(*) as null_count FROM `{self.table_name}` WHERE `{col_name}` IS NULL"
                                        null_count = pd.read_sql_query(null_count_query, conn).iloc[0]['null_count']
                                        
                                        for sample in date_samples['sample_date'].tolist():
                                            if sample:
                                                formats.append(str(sample))
                                        metadata['date_formats'][col_name] = {
                                            'examples': formats[:6],
                                            'null_count': null_count,
                                            'null_percentage': round((null_count / metadata['row_count']) * 100, 1)
                                        }
                                else:
                                    # Analyze as categorical column
                                    unique_count_query = f"SELECT COUNT(DISTINCT `{col_name}`) as unique_count FROM `{self.table_name}` WHERE `{col_name}` IS NOT NULL"
                                    unique_count = pd.read_sql_query(unique_count_query, conn).iloc[0]['unique_count']
                                    
                                    # Get null count
                                    null_count_query = f"SELECT COUNT(*) as null_count FROM `{self.table_name}` WHERE `{col_name}` IS NULL"
                                    null_count = pd.read_sql_query(null_count_query, conn).iloc[0]['null_count']
                                    
                                    # Fetch unique values based on cardinality
                                    if unique_count <= 100:  # Full categorical analysis for low cardinality
                                        limit = 25 if is_priority else 15
                                        unique_values_query = f"""
                                        SELECT `{col_name}`, COUNT(*) as count 
                                        FROM `{self.table_name}` 
                                        WHERE `{col_name}` IS NOT NULL 
                                        GROUP BY `{col_name}` 
                                        ORDER BY count DESC 
                                        LIMIT {limit}
                                        """
                                        unique_values = pd.read_sql_query(unique_values_query, conn)
                                        if not unique_values.empty:
                                            values_with_counts = []
                                            for _, row in unique_values.iterrows():
                                                values_with_counts.append({
                                                    'value': row[col_name],
                                                    'count': row['count']
                                                })
                                            metadata['categorical_values'][col_name] = {
                                                'total_unique': unique_count,
                                                'null_count': null_count,
                                                'null_percentage': round((null_count / metadata['row_count']) * 100, 1),
                                                'top_values': values_with_counts,
                                                'is_high_cardinality': unique_count > 50
                                            }
                                    elif unique_count <= 1000 and is_priority:  # Sample for medium cardinality priority columns
                                        sample_values_query = f"""
                                        SELECT `{col_name}`, COUNT(*) as count 
                                        FROM `{self.table_name}` 
                                        WHERE `{col_name}` IS NOT NULL 
                                        GROUP BY `{col_name}` 
                                        ORDER BY count DESC 
                                        LIMIT 10
                                        """
                                        sample_values = pd.read_sql_query(sample_values_query, conn)
                                        if not sample_values.empty:
                                            values_with_counts = []
                                            for _, row in sample_values.iterrows():
                                                values_with_counts.append({
                                                    'value': row[col_name],
                                                    'count': row['count']
                                                })
                                            metadata['categorical_values'][col_name] = {
                                                'total_unique': unique_count,
                                                'null_count': null_count,
                                                'null_percentage': round((null_count / metadata['row_count']) * 100, 1),
                                                'top_values': values_with_counts,
                                                'is_high_cardinality': True,
                                                'note': f'High cardinality column - showing top 10 of {unique_count} values'
                                            }
                            
                            # For REAL/INTEGER columns, get basic statistics
                            elif col_type.upper() in ['REAL', 'INTEGER']:
                                if should_analyze:
                                    stats_query = f"""
                                    SELECT 
                                        MIN(`{col_name}`) as min_val,
                                        MAX(`{col_name}`) as max_val,
                                        AVG(`{col_name}`) as avg_val,
                                        COUNT(`{col_name}`) as non_null_count,
                                        COUNT(*) - COUNT(`{col_name}`) as null_count
                                    FROM `{self.table_name}`
                                    """
                                    stats = pd.read_sql_query(stats_query, conn)
                                    if not stats.empty:
                                        stat_row = stats.iloc[0]
                                        metadata['numeric_stats'][col_name] = {
                                            'min': stat_row['min_val'],
                                            'max': stat_row['max_val'],
                                            'avg': round(float(stat_row['avg_val']), 2) if stat_row['avg_val'] is not None else None,
                                            'non_null_count': stat_row['non_null_count'],
                                            'null_count': stat_row['null_count'],
                                            'null_percentage': round((stat_row['null_count'] / metadata['row_count']) * 100, 1)
                                        }
                            
                            analyzed_count += 1
                            
                        except Exception as e:
                            print(f"⚠️ Warning: Could not analyze column '{col_name}': {str(e)}")
                            continue
                
                print(f"✅ Dynamic metadata fetched for {len(metadata['columns'])} columns")
                print(f"📊 Detailed analysis: {len(metadata['categorical_values'])} categorical, {len(metadata['date_formats'])} date, {len(metadata['numeric_stats'])} numeric columns")
                return metadata
                
        except Exception as e:
            print(f"❌ Error fetching dynamic metadata: {str(e)}")
            return metadata
    
    def _extract_columns_from_query(self, user_question):
        """
        Extract potential column names from user question to prioritize metadata fetching
        """
        if not user_question:
            return []
        
        question_lower = user_question.lower()
        potential_columns = []
        
        # Direct column name mapping
        column_keywords = {
            'complaint': ['Complaint Type', 'Descriptor'],
            'type': ['Complaint Type'],
            'status': ['Status'],
            'agency': ['Agency', 'Agency Name'],
            'location': ['Location Type', 'Borough', 'City', 'Incident Address'],
            'address': ['Incident Address', 'Street Name'],
            'date': ['Created Date', 'Closed Date', 'Due Date'],
            'time': ['Created Date', 'Closed Date'],
            'created': ['Created Date'],
            'closed': ['Closed Date'],
            'due': ['Due Date'],
            'borough': ['Borough'],
            'city': ['City'],
            'zip': ['Incident Zip'],
            'latitude': ['Latitude'],
            'longitude': ['Longitude'],
            'coordinate': ['Latitude', 'Longitude', 'X Coordinate (State Plane)', 'Y Coordinate (State Plane)'],
            'resolution': ['Resolution Description', 'Resolution Action Updated Date'],
            'community': ['Community Board'],
            'noise': ['Complaint Type'],
            'parking': ['Complaint Type'],
            'street': ['Complaint Type', 'Street Name'],
            'vehicle': ['Complaint Type', 'Vehicle Type'],
            'school': ['School Name', 'School Number', 'School Region'],
            'bridge': ['Bridge Highway Name', 'Bridge Highway Direction'],
            'ferry': ['Ferry Direction', 'Ferry Terminal Name']
        }
        
        # Look for keywords in the question
        for keyword, columns in column_keywords.items():
            if keyword in question_lower:
                potential_columns.extend(columns)
        
        # Look for specific complaint types mentioned
        complaint_type_keywords = {
            'noise': 'Complaint Type',
            'parking': 'Complaint Type', 
            'blocked': 'Complaint Type',
            'illegal': 'Complaint Type',
            'driveway': 'Complaint Type',
            'vehicle': 'Complaint Type',
            'animal': 'Complaint Type',
            'traffic': 'Complaint Type'
        }
        
        for keyword, column in complaint_type_keywords.items():
            if keyword in question_lower and column not in potential_columns:
                potential_columns.append(column)
        
        # Look for temporal analysis keywords
        temporal_keywords = ['trend', 'over time', 'year', 'month', 'daily', 'weekly', 'timeline', 'history']
        if any(keyword in question_lower for keyword in temporal_keywords):
            if 'Created Date' not in potential_columns:
                potential_columns.append('Created Date')
        
        # Look for geographic analysis keywords  
        geographic_keywords = ['location', 'where', 'borough', 'area', 'map', 'geographic', 'spatial']
        if any(keyword in question_lower for keyword in geographic_keywords):
            geo_columns = ['Borough', 'City', 'Latitude', 'Longitude']
            potential_columns.extend([col for col in geo_columns if col not in potential_columns])
        
        # Remove duplicates and return
        unique_columns = list(set(potential_columns))
        if unique_columns:
            print(f"🎯 Query-specific columns identified: {unique_columns}")
        
        return unique_columns
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(SQLAgentState)
        
        # Add nodes
        workflow.add_node("check_if_clarification_needed", self._check_if_clarification_needed)
        workflow.add_node("generate_query", self._generate_sql_query)
        workflow.add_node("execute_query", self._execute_sql_query)
        workflow.add_node("handle_error", self._handle_query_error)
        workflow.add_node("generate_visualization", self._generate_visualization)
        workflow.add_node("generate_answer", self._generate_final_answer)
        
        # Add edges
        workflow.set_entry_point("check_if_clarification_needed")
        
        workflow.add_conditional_edges(
            "check_if_clarification_needed",
            self._should_ask_for_clarification,
            {
                "clarification_needed": END,
                "proceed": "generate_query"
            }
        )
        
        workflow.add_edge("generate_query", "execute_query")
        
        workflow.add_conditional_edges(
            "execute_query",
            self._should_retry_or_finish,
            {
                "retry": "handle_error",
                "success": "generate_visualization",
                "max_attempts": "generate_visualization"
            }
        )
        
        workflow.add_edge("handle_error", "generate_query")
        workflow.add_edge("generate_visualization", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
        
    def _check_if_clarification_needed(self, state: SQLAgentState) -> SQLAgentState:
        """Check if we need to ask for clarification from the user"""
        print("🔄 Checking if clarification is needed...")
        
        # If we don't have a conversation history or it's just the welcome message, we don't need clarification
        if not state.get("conversation_history") or len(state["conversation_history"]) <= 1:
            return {
                **state,
                "requires_clarification": False
            }
        
        # Format the conversation history for the LLM - use recent messages for context
        conversation_context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in state["conversation_history"][-5:] # Use recent 5 messages for context
        ])
        
        system_prompt = f"""You are an expert SQL query planner for NYC 311 service request data. Your job is to determine if a user's question is clear enough to write a SQL query, or if you need clarification.

DATABASE SCHEMA:
{state['table_schema']}

AVAILABLE COLUMNS FOR REFERENCE:
{chr(10).join([f"- {col['name']} ({col['type']})" for _, col in self.table_columns.iterrows()])}

CONVERSATION HISTORY:
{conversation_context}

INSTRUCTIONS:
1. **Context Awareness**: Analyze the user's latest question in the full context of the conversation history
2. **Follow-up Recognition**: If the current question is a follow-up to a previous query (like "make a scatter plot" after getting complaint data), understand the connection
3. **Data Availability**: Check if you can reasonably interpret the request based on previous queries and available columns
4. **Smart Suggestions**: For visualization requests, consider data from previous queries or suggest meaningful column combinations
5. **Minimal Clarification**: Only ask for clarification if absolutely necessary - try to provide intelligent defaults based on context

CONTEXT-AWARE DECISION MAKING:
- If user asks for a different visualization type (scatter, pie, bar) after a data query, assume they want to visualize related data
- Consider the conversation flow and previous data requests
- Provide helpful suggestions rather than asking open-ended questions

If clarification is needed, your response should be in this format:
```
NEEDS_CLARIFICATION: true
CLARIFICATION_QUESTION: Your specific clarification question here, mentioning relevant available columns
```

If no clarification is needed:
```
NEEDS_CLARIFICATION: false
```

EXAMPLES OF WHEN CLARIFICATION IS NEEDED:
- User asks for "top complaints" without specifying a number or time period (ONLY if not clear from context)
- User refers to "it" or "them" without clear antecedents AND cannot be inferred from conversation
- User asks for completely new analysis without sufficient context AND available columns don't suggest obvious options
- User asks about data that clearly doesn't exist in the database

EXAMPLES OF WHEN CLARIFICATION IS NOT NEEDED:
- User asks for different visualization after a data query (suggest reasonable defaults)
- User asks for "scatter plot" - suggest meaningful column pairs like Latitude/Longitude or time-based analysis
- User asks for follow-up analysis that can be reasonably inferred from previous queries
- User requests are contextually clear from conversation history

CLARIFICATION STRATEGY:
- Provide intelligent suggestions rather than asking open-ended questions
- Reference previous conversation context
- Suggest specific meaningful column combinations
- Only clarify when absolutely necessary
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User's latest question: {state['user_question']}")
        ]
        
        # Use moderate timeout for clarification check
        response = self.client.invoke(messages, timeout=90, max_tokens=2000)
        response_text = response.content
        
        # Parse the response to check if clarification is needed
        needs_clarification = "NEEDS_CLARIFICATION: true" in response_text
        clarification_question = ""
        
        if needs_clarification:
            # Extract the clarification question
            for line in response_text.split("\n"):
                if "CLARIFICATION_QUESTION:" in line:
                    clarification_question = line.split("CLARIFICATION_QUESTION:", 1)[1].strip()
                    break
        
        if needs_clarification and clarification_question:
            print(f" Clarification needed: {clarification_question}")
            return {
                **state,
                "requires_clarification": True,
                "clarification_question": clarification_question,
                "final_answer": f"<div class='clarification-request'><p>I need a bit more information to give you an accurate answer. {clarification_question}</p></div>"
            }
        else:
            print("✅ No clarification needed, proceeding with query generation")
            return {
                **state,
                "requires_clarification": False,
                "clarification_question": ""
            }
            
    def _should_ask_for_clarification(self, state: SQLAgentState) -> str:
        """Conditional routing based on clarification needs"""
        if state["requires_clarification"]:
            return "clarification_needed"
        else:
            return "proceed"
    
    def _generate_sql_query(self, state: SQLAgentState) -> SQLAgentState:
        """Generate SQL query"""
        print(f"\n🔄 Generating SQL query (attempt {state['attempt_count'] + 1})...")
        
        # Extract potential column names from user query for targeted analysis
        query_specific_columns = self._extract_columns_from_query(state['user_question'])
        
        # Fetch dynamic metadata with priority on query-relevant columns
        dynamic_metadata = self._get_dynamic_table_metadata(query_specific_columns)
        
        # Prepare conversation context with recent history for follow-up handling
        conversation_context = ""
        if state.get("conversation_history") and len(state["conversation_history"]) > 1:
            # Use recent 5 messages to maintain context for follow-up queries
            conversation_messages = state["conversation_history"][-5:] if len(state["conversation_history"]) > 5 else state["conversation_history"]
            conversation_context = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_messages
            ])
            conversation_context = f"\nCONVERSATION HISTORY (Use this context to understand follow-up requests):\n{conversation_context}\n"
        
        # Build dynamic categorical values section
        categorical_section = ""
        if dynamic_metadata['categorical_values']:
            categorical_section = "\n🎯 VALID VALUES FOR CATEGORICAL COLUMNS (Use EXACT values - case sensitive):\n"
            for col_name, col_data in dynamic_metadata['categorical_values'].items():
                if col_data['top_values']:
                    null_info = f" | {col_data['null_percentage']}% NULL" if col_data['null_percentage'] > 0 else ""
                    cardinality_note = " | HIGH CARDINALITY" if col_data.get('is_high_cardinality') else ""
                    categorical_section += f"\n`{col_name}` ({col_data['total_unique']} unique values{null_info}{cardinality_note}):\n"
                    
                    for i, value_info in enumerate(col_data['top_values'], 1):
                        categorical_section += f"  {i}. \"{value_info['value']}\" ({value_info['count']:,} records)\n"
                    
                    if col_data['total_unique'] > len(col_data['top_values']):
                        remaining = col_data['total_unique'] - len(col_data['top_values'])
                        categorical_section += f"  ... and {remaining} more values\n"
                    
                    if col_data.get('note'):
                        categorical_section += f"  {col_data['note']}\n"
            
            # Add mapping warnings for common mistakes  
            if any('Complaint Type' in col for col in dynamic_metadata['categorical_values'].keys()):
                categorical_section += """
 IMPORTANT MAPPING RULES FOR USER QUERIES:
- "Noise complaints" → Use: "Noise - Street/Sidewalk", "Noise - Commercial", "Noise - Vehicle", "Noise - Park" 
- "Street condition" → Check available values above - no exact "Street Condition" type exists
- "Parking complaints" → Use: "Illegal Parking", "Blocked Driveway"
- "Traffic issues" → Use: "Traffic", "Derelict Vehicle"
- Always use EXACT values from the list above - case and punctuation sensitive!
- When user asks about broad categories, include ALL relevant specific types in your query
"""
        
        # Build date formats section
        date_formats_section = ""
        if dynamic_metadata['date_formats']:
            date_formats_section = "\n📅 DATE COLUMN FORMATS (Actual examples from data):\n"
            for col_name, date_info in dynamic_metadata['date_formats'].items():
                null_info = f" | {date_info['null_percentage']}% NULL ({date_info['null_count']:,} records)" if date_info['null_percentage'] > 0 else ""
                date_formats_section += f"\n`{col_name}`{null_info}:\n"
                for fmt in date_info['examples']:
                    date_formats_section += f"  - \"{fmt}\"\n"
                if date_info['null_percentage'] > 0:
                    date_formats_section += f"   NULL dates mean 'not yet occurred' or 'still pending'\n"
        
        # Build numeric statistics section
        numeric_section = ""
        if dynamic_metadata['numeric_stats']:
            numeric_section = "\n NUMERIC COLUMN STATISTICS:\n"
            for col_name, stats in dynamic_metadata['numeric_stats'].items():
                null_info = f" | {stats['null_percentage']}% NULL" if stats['null_percentage'] > 0 else ""
                numeric_section += f"\n`{col_name}`{null_info}:\n"
                numeric_section += f"  Range: {stats['min']} to {stats['max']}\n"
                if stats['avg'] is not None:
                    numeric_section += f"  Average: {stats['avg']}\n"
                numeric_section += f"  Non-null records: {stats['non_null_count']:,}\n"
        
        # Prepare context for LLM
        system_prompt = f"""You are an expert SQL query generator specializing in NYC 311 service request data analysis. Generate precise SQL queries that capture comprehensive numerical insights.

DATABASE SCHEMA:
{state['table_schema']}
 DATASET OVERVIEW:
- Table: `{self.table_name}`
- Total Records: {dynamic_metadata['row_count']:,}
- Total Columns: {len(self.table_columns)}

AVAILABLE COLUMNS (USE ONLY THESE - DO NOT USE ANY OTHER COLUMN NAMES):
{chr(10).join([f"- `{col['name']}` ({col['type']})" for _, col in self.table_columns.iterrows()])}
{categorical_section}
{date_formats_section}
{numeric_section}

COLUMN VALIDATION CHECKLIST:
✓ Total columns available: {len(self.table_columns)}
✓ Table name: `{self.table_name}` (always use backticks)
✓ All column names MUST be wrapped in backticks if they contain spaces
✓ Geographic columns: `Latitude`, `Longitude`, `Borough`, `City`
✓ Temporal columns: `Created Date`, `Closed Date` (TEXT format - use JULIANDAY for calculations)

{conversation_context}

CRITICAL REQUIREMENTS - COLUMN NAME VALIDATION:
1. **MANDATORY COLUMN CHECK**: Before writing ANY SQL query, you MUST verify that EVERY column name you use exists in the available columns list above
2. **EXACT COLUMN NAMES ONLY**: Use column names EXACTLY as they appear in the list - case-sensitive with exact spacing
3. **STRICT NO-INVENTION POLICY**: NEVER create, assume, invent, or guess column names not explicitly listed
4. **COMMON MISTAKE PREVENTION**: 
   - DO NOT use variations like "complaint_type" when column is "Complaint Type"
   - DO NOT use "agency_name" when column is "Agency"
   - DO NOT use "created_at" when column is "Created Date"
   - DO NOT use "location_type" when column is "Location Type"
5. **BACKTICK REQUIREMENT**: Always wrap column names containing spaces in backticks: `Complaint Type`
6. **TABLE NAME**: Always use `{self.table_name}` with backticks
7. **ERROR PREVENTION**: If user asks about non-existent columns, respond with available alternatives from the list
8. **SYNTAX COMPLIANCE**: Use only SQLite-compatible syntax with proper escaping

COLUMN NAME CROSS-REFERENCE (Use ONLY these exact names):
Geographic: `Borough`, `City`, `Latitude`, `Longitude`, `Location Type`
Temporal: `Created Date`, `Closed Date`
Categorical: `Complaint Type`, `Agency`, `Status`, `Descriptor`
Administrative: `Unique Key`, `Agency Name`, `Community Board`

QUERY OPTIMIZATION RULES:
1. For analytical queries, include relevant statistical measures (COUNT, SUM, AVG, MIN, MAX, percentages)
2. Add calculated fields for percentages, ratios, and rankings when appropriate
3. Use proper SQL syntax for SQLite with optimized performance
4. Include meaningful column aliases for better readability
5. Order results by most significant metrics (usually counts or percentages DESC)
6. If visualization is expected, limit results appropriately (TOP 10-20 for charts)

DATA CONTEXT: NYC 311 Service Request Database
- Contains citizen service requests, complaints, and inquiries  
- Key metrics: request volumes, geographic distributions, agency performance
- Time-sensitive data with `Created Date` and `Closed Date` columns (TEXT format)
- Many records may have NULL `Closed Date` (open/pending requests)
- Use JULIANDAY() function for date calculations and comparisons

CRITICAL NULL DATE HANDLING RULES:
NULL `Closed Date` = COMPLAINT IS STILL OPEN/PENDING (NOT CLOSED YET)

1. **For Percentage Calculations (MOST IMPORTANT):**
   - When user asks "What percentage closed within X timeframe?"
   - DENOMINATOR = COUNT(*) [ALL complaints including NULL dates]
   - NUMERATOR = COUNT(complaints with valid close dates within timeframe)
   - NULL dates should NEVER be excluded from total count
   - NULL dates mean "NOT closed within any timeframe"

2. **For Closure Analysis:**
   - Always show three categories:
     * Closed within timeframe (has close date ≤ X days)
     * Closed after timeframe (has close date > X days)  
     * Still open/pending (NULL close date)

3. **Correct Query Pattern for "% closed within X days":**
```sql
SELECT
    COUNT(*) as total_complaints,
    COUNT(CASE WHEN `Closed Date` IS NOT NULL AND date_diff <= X THEN 1 END) as closed_within_X_days,
    COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) as still_open,
    COUNT(CASE WHEN `Closed Date` IS NOT NULL AND date_diff > X THEN 1 END) as closed_after_X_days,
    ROUND(COUNT(CASE WHEN `Closed Date` IS NOT NULL AND date_diff <= X THEN 1 END) * 100.0 / COUNT(*), 2) as percentage_closed_within_X_days,
    ROUND(COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) * 100.0 / COUNT(*), 2) as percentage_still_open
FROM table_name;
```

4. **NEVER Do This (Common Mistake):**
 WHERE `Closed Date` IS NOT NULL  -- This excludes open complaints from analysis
 Using only closed cases in percentage denominator

5. **Always Do This:**
 Include NULL dates in total count
 Report open/pending cases separately
 Use meaningful labels like "Still Open" or "Pending"

DEBUGGING TIPS:
- If a query returns 0 rows, try removing some WHERE conditions
- For date analysis, first check data availability with: SELECT COUNT(*) FROM `{self.table_name}` WHERE `Closed Date` IS NOT NULL
- Use LIMIT to prevent overwhelming results

PRE-QUERY VALIDATION CHECKLIST:
Before writing ANY SQL query, you MUST:
✓ Read the user's question carefully
✓ Identify what columns you need to answer the question
✓ Check that EVERY column exists in the available columns list above
✓ Use exact column names from the list (case-sensitive, exact spacing)
✓ Wrap column names with spaces in backticks
✓ If a needed column doesn't exist, suggest alternatives from available columns

QUERY GENERATION INSTRUCTIONS:
1. **Context Awareness**: If the current question is a short response (like "days", "hours", "yes") or a visualization request (like "make a scatter plot"), look at the conversation history to understand the original question and context
2. **MANDATORY Column Validation**: Before writing any query, perform the pre-query validation checklist above
3. **Follow-up Visualization Requests**: When user asks for a different chart type after a data query, understand they want to visualize related or meaningful data
4. **Smart Defaults for Scatter Plots**: For scatter plot requests, suggest meaningful numeric column pairs like:
   - `Latitude` vs `Longitude` (geographic distribution)
   - Date-based analysis using JULIANDAY calculations
   - Numeric columns that show relationships
5. **Follow-up Handling**: When user provides clarification or requests new visualizations, combine with context from previous queries
6. **Column Name Safety**: If you're unsure about a column name, refer back to the exact list provided above

ENHANCED QUERY PATTERNS (using actual column names):
- Basic counts: SELECT `[actual_column_name]`, COUNT(*) as total_requests FROM `{self.table_name}` GROUP BY `[actual_column_name]` ORDER BY total_requests DESC LIMIT 10
- With percentages: Add ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM `{self.table_name}`), 2) as percentage
- Status breakdown: SELECT `Status`, COUNT(*) FROM `{self.table_name}` GROUP BY `Status`

SPECIFIC PATTERN FOR TOP COMPLAINTS WITH CLOSURE TIMEFRAME:
For "top X complaints with Y% closed within Z months":
CORRECT APPROACH - Include NULL dates in total count:
```sql
WITH top_complaints AS (
    SELECT `Complaint Type`, COUNT(*) as total_requests
    FROM `{self.table_name}`
    WHERE `Complaint Type` IS NOT NULL
    GROUP BY `Complaint Type`
    ORDER BY COUNT(*) DESC
    LIMIT 5
)
SELECT 
    tc.`Complaint Type`,
    tc.total_requests,  -- This is the TRUE total (includes NULL closed dates)
    COUNT(CASE 
        WHEN t.`Closed Date` IS NOT NULL 
         AND t.`Created Date` IS NOT NULL
         AND JULIANDAY(SUBSTR(t.`Closed Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(t.`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(t.`Closed Date`, 4, 2) AS INTEGER))) -
             JULIANDAY(SUBSTR(t.`Created Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(t.`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(t.`Created Date`, 4, 2) AS INTEGER))) <= 270
        THEN 1 END) as closed_within_9_months,
    COUNT(CASE WHEN t.`Closed Date` IS NULL THEN 1 END) as still_open,
    COUNT(CASE 
        WHEN t.`Closed Date` IS NOT NULL 
         AND t.`Created Date` IS NOT NULL
         AND JULIANDAY(SUBSTR(t.`Closed Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(t.`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(t.`Closed Date`, 4, 2) AS INTEGER))) -
             JULIANDAY(SUBSTR(t.`Created Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(t.`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(t.`Created Date`, 4, 2) AS INTEGER))) > 270
        THEN 1 END) as closed_after_9_months,
    -- CORRECT PERCENTAGE: Uses total_requests (includes NULL) as denominator
    ROUND(COUNT(CASE 
        WHEN t.`Closed Date` IS NOT NULL 
         AND t.`Created Date` IS NOT NULL
         AND JULIANDAY(SUBSTR(t.`Closed Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(t.`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(t.`Closed Date`, 4, 2) AS INTEGER))) -
             JULIANDAY(SUBSTR(t.`Created Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(t.`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(t.`Created Date`, 4, 2) AS INTEGER))) <= 270
        THEN 1 END) * 100.0 / tc.total_requests, 2) as percentage_closed_within_9_months,
    ROUND(COUNT(CASE WHEN t.`Closed Date` IS NULL THEN 1 END) * 100.0 / tc.total_requests, 2) as percentage_still_open
FROM `{self.table_name}` t
JOIN top_complaints tc ON t.`Complaint Type` = tc.`Complaint Type`
GROUP BY tc.`Complaint Type`, tc.total_requests
ORDER BY tc.total_requests DESC;
```

DATE FORMAT HANDLING - CRITICAL FOR ACCURACY:
The database contains dates in mixed formats:
- Format 1: "12/31/2015 09:04:05 PM" (MM/DD/YYYY HH:MM:SS AM/PM)
- Format 2: "01-01-2016 07:43" (MM-DD-YYYY HH:MM)

SAFE DATE PARSING STRATEGY:
For any date calculations, use this robust approach that properly handles NULL dates:

CRITICAL: NULL `Closed Date` = NOT CLOSED YET (include in analysis)

```sql
-- For timeframe analysis with NULL handling:
SELECT
    COUNT(*) as total_complaints,  -- Always include ALL complaints
    COUNT(CASE 
        WHEN `Closed Date` IS NOT NULL 
         AND `Created Date` IS NOT NULL
         AND `Closed Date` LIKE '%/%' 
         AND `Created Date` LIKE '%/%'
         AND CAST((JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                   SUBSTR(`Closed Date`, 1, 2) || '-' || 
                   SUBSTR(`Closed Date`, 4, 2)) - 
                  JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' || 
                   SUBSTR(`Created Date`, 1, 2) || '-' || 
                   SUBSTR(`Created Date`, 4, 2))) AS INTEGER) <= X_DAYS
        THEN 1 END) as closed_within_timeframe,
    COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) as still_open,
    COUNT(CASE 
        WHEN `Closed Date` IS NOT NULL 
         AND `Created Date` IS NOT NULL
         AND `Closed Date` LIKE '%/%' 
         AND `Created Date` LIKE '%/%'
         AND CAST((JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                   SUBSTR(`Closed Date`, 1, 2) || '-' || 
                   SUBSTR(`Closed Date`, 4, 2)) - 
                  JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' || 
                   SUBSTR(`Created Date`, 1, 2) || '-' || 
                   SUBSTR(`Created Date`, 4, 2))) AS INTEGER) > X_DAYS
        THEN 1 END) as closed_after_timeframe,
    -- Percentage calculations using total count as denominator
    ROUND(COUNT(CASE 
        WHEN `Closed Date` IS NOT NULL AND date_calculation <= X_DAYS
        THEN 1 END) * 100.0 / COUNT(*), 2) as percentage_closed_within_timeframe,
    ROUND(COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) * 100.0 / COUNT(*), 2) as percentage_still_open
FROM table_name;

-- For mixed date formats (robust version):
CASE 
    WHEN `Closed Date` IS NULL THEN 'OPEN'  -- Handle NULL explicitly
    WHEN `Closed Date` LIKE '%/%' AND `Created Date` LIKE '%/%' THEN
        CASE 
            WHEN CAST((JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                      SUBSTR(`Closed Date`, 1, 2) || '-' || 
                      SUBSTR(`Closed Date`, 4, 2)) - 
                     JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' || 
                      SUBSTR(`Created Date`, 1, 2) || '-' || 
                      SUBSTR(`Created Date`, 4, 2))) AS INTEGER) <= X_DAYS
            THEN 'CLOSED_WITHIN_TIMEFRAME'
            ELSE 'CLOSED_AFTER_TIMEFRAME'
        END
    WHEN `Closed Date` LIKE '%-%' AND `Created Date` LIKE '%-%' THEN
        CASE 
            WHEN CAST((JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                      SUBSTR(`Closed Date`, 1, 2) || '-' || 
                      SUBSTR(`Closed Date`, 4, 2)) - 
                     JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' || 
                      SUBSTR(`Created Date`, 1, 2) || '-' || 
                      SUBSTR(`Created Date`, 4, 2))) AS INTEGER) <= X_DAYS
            THEN 'CLOSED_WITHIN_TIMEFRAME'
            ELSE 'CLOSED_AFTER_TIMEFRAME'
        END
    ELSE 'INVALID_DATE_FORMAT'
END
```
```

SIMPLIFIED DATE CALCULATIONS FOR TIMEFRAME QUERIES:
REMEMBER: NULL `Closed Date` = STILL OPEN (never exclude from totals)

For queries involving timeframes (like "within 9 months" = 270 days):

CORRECT PATTERN - Always count NULL dates:
```sql
SELECT
    COUNT(*) as total_complaints,  -- TRUE TOTAL: includes NULL closed dates
    COUNT(CASE 
        WHEN `Closed Date` IS NOT NULL 
         AND `Created Date` IS NOT NULL
         AND JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 4, 2) AS INTEGER))) -
             JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' || 
                      PRINTF('%02d', CAST(SUBSTR(`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                      PRINTF('%02d', CAST(SUBSTR(`Created Date`, 4, 2) AS INTEGER))) <= X_DAYS
        THEN 1 END) as closed_within_timeframe,
    COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) as still_open,
    -- CORRECT PERCENTAGE: NULL dates counted in denominator
    ROUND(COUNT(CASE WHEN `Closed Date` IS NOT NULL AND date_calc <= X_DAYS THEN 1 END) * 100.0 / COUNT(*), 2) as percentage
FROM table_name;
```

WRONG APPROACH (common mistake):
```sql
-- DON'T DO THIS - excludes open complaints:
WHERE `Closed Date` IS NOT NULL  
-- DON'T DO THIS - wrong denominator:
COUNT(closed_cases) * 100.0 / COUNT(closed_cases_only)
```
RIGHT APPROACH:
```sql
-- Always include NULL dates in analysis
-- Use COUNT(*) as denominator for percentages
-- Report NULL cases as "Still Open" or "Pending"
```

BETTER APPROACH - Use DATETIME() function with proper format conversion:
```sql
-- For 9 months (270 days) check:
WHERE DATETIME(SUBSTR(`Closed Date`, 7, 4) || '-' || 
               PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
               PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 4, 2) AS INTEGER))) IS NOT NULL
  AND DATETIME(SUBSTR(`Created Date`, 7, 4) || '-' || 
               PRINTF('%02d', CAST(SUBSTR(`Created Date`, 1, 2) AS INTEGER)) || '-' ||
               PRINTF('%02d', CAST(SUBSTR(`Created Date`, 4, 2) AS INTEGER))) IS NOT NULL
  AND JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 4, 2) AS INTEGER))) -
      JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' || 
                PRINTF('%02d', CAST(SUBSTR(`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                PRINTF('%02d', CAST(SUBSTR(`Created Date`, 4, 2) AS INTEGER))) <= 270
```

CRITICAL DATE HANDLING RULES:
1. **Never assume date format** - always check for both "/" and "-" separators
2. **Use SUBSTR() to extract parts** - Year (position 7, length 4), Month (position 1, length 2), Day (position 4, length 2)
3. **Convert to ISO format** - YYYY-MM-DD before using JULIANDAY()
4. **Test date validity** - Use IS NOT NULL checks after conversion
5. **Handle mixed formats** - Use CASE statements for different format detection

COLUMN NAME VALIDATION - USE ONLY THESE EXACT NAMES:
{chr(10).join([f"✓ `{col['name']}`" for _, col in self.table_columns.iterrows()])}

EXAMPLE: "What percentage of complaints were closed within 4 days?"
CORRECT APPROACH - Include NULL dates in total count:
```sql
SELECT
    COUNT(*) as total_complaints,
    COUNT(CASE WHEN `Closed Date` IS NOT NULL AND 
               JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                        PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                        PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 4, 2) AS INTEGER))) -
               JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' ||
                        PRINTF('%02d', CAST(SUBSTR(`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                        PRINTF('%02d', CAST(SUBSTR(`Created Date`, 4, 2) AS INTEGER))) <= 4
          THEN 1 END) as closed_within_4_days,
    COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) as still_open,
    ROUND(COUNT(CASE WHEN `Closed Date` IS NOT NULL AND 
                     JULIANDAY(SUBSTR(`Closed Date`, 7, 4) || '-' || 
                              PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 1, 2) AS INTEGER)) || '-' ||
                              PRINTF('%02d', CAST(SUBSTR(`Closed Date`, 4, 2) AS INTEGER))) -
                     JULIANDAY(SUBSTR(`Created Date`, 7, 4) || '-' ||
                              PRINTF('%02d', CAST(SUBSTR(`Created Date`, 1, 2) AS INTEGER)) || '-' ||
                              PRINTF('%02d', CAST(SUBSTR(`Created Date`, 4, 2) AS INTEGER))) <= 4
                THEN 1 END) * 100.0 / COUNT(*), 2) as percentage_closed_within_4_days,
    ROUND(COUNT(CASE WHEN `Closed Date` IS NULL THEN 1 END) * 100.0 / COUNT(*), 2) as percentage_still_open
FROM `{self.table_name}`;
```

IMPORTANT: 
- Use these exact column names (with exact spacing and capitalization) in backticks
- For calculated fields, repeat the full calculation in GROUP BY/ORDER BY, don't use aliases

"""
        
        # Add error context if this is a retry
        if state.get('error_message'):
            system_prompt += f"\nPREVIOUS ERROR: {state['error_message']}\nPREVIOUS QUERY: {state.get('sql_query', 'None')}\nPlease fix the query based on the error."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Question: {state['user_question']}")
        ]
        
        # Use longer timeout for SQL query generation (complex prompts)
        response = self.client.invoke(messages, timeout=120, max_tokens=3000)
        
        # Extract SQL query from response
        sql_query = response.content.strip()
        
        # Clean up the query (remove markdown formatting if present)
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        print(f"📝 Generated query: {sql_query}")
        
        return {
            **state,
            "sql_query": sql_query,
            "attempt_count": state["attempt_count"] + 1
        }
    
    def _execute_sql_query(self, state: SQLAgentState) -> SQLAgentState:
        """Execute the SQL query"""
        print("⚡ Executing SQL query...")
        
        try:
            result = self._query_database(state["sql_query"])
            
            if result is not None:
                print(f"✅ Query executed successfully! Returned {len(result)} rows")
                return {
                    **state,
                    "query_result": result,
                    "error_message": None
                }
            else:
                return {
                    **state,
                    "query_result": None,
                    "error_message": "Query returned None - check the query syntax"
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Query execution failed: {error_msg}")
            return {
                **state,
                "query_result": None,
                "error_message": error_msg
            }
    
    def _should_retry_or_finish(self, state: SQLAgentState) -> str:
        """Determine whether to retry, finish successfully, or stop at max attempts"""
        if state["query_result"] is not None:
            return "success"
        elif state["attempt_count"] >= state["max_attempts"]:
            return "max_attempts"
        else:
            return "retry"
    
    def _handle_query_error(self, state: SQLAgentState) -> SQLAgentState:
        """Handle query errors and prepare for retry"""
        print(f"🔄 Handling error for retry {state['attempt_count']}/{state['max_attempts']}")
        return state  # State is passed through to retry query generation
    
    def _generate_visualization(self, state: SQLAgentState) -> SQLAgentState:
        """Generate visualization using LLM to create Python matplotlib code and generate image with retry logic"""
        print("📊 Generating LLM-based visualization with image generation...")
        
        query_result = state.get("query_result")
        user_question = state.get("user_question", "")
        
        visualization_data = None
        max_code_attempts = 3  # Maximum retry attempts for code execution
        
        if query_result is not None and not query_result.empty:
            print("🔍 Asking LLM to analyze data and create visualization...")
            
            # Prepare data summary for LLM
            data_summary = f"""
DATA OVERVIEW:
- Rows: {len(query_result)}
- Columns: {list(query_result.columns)}
- Sample data (first 5 rows):
{query_result.head().to_string(index=False)}

FULL DATASET:
{query_result.to_string(index=False, max_rows=50)}
"""
            
            # Retry loop for code generation and execution
            for attempt in range(1, max_code_attempts + 1):
                print(f"📊 Code generation attempt {attempt}/{max_code_attempts}")
                
                try:
                    visualization_data = self._generate_and_execute_visualization_code(
                        user_question, data_summary, attempt, 
                        state.get("previous_code_errors", [])
                    )
                    
                    if visualization_data and visualization_data.get("type") == "image":
                        print(f"✅ Successfully generated visualization on attempt {attempt}")
                        break
                    else:
                        raise Exception("Generated code did not produce valid visualization")
                        
                except Exception as e:
                    print(f"❌ Attempt {attempt} failed: {e}")
                    
                    # Store error for next attempt
                    if "previous_code_errors" not in state:
                        state["previous_code_errors"] = []
                    state["previous_code_errors"].append({
                        "attempt": attempt,
                        "error": str(e),
                        "timestamp": str(datetime.now()) if 'datetime' in globals() else "unknown"
                    })
                    
                    # If this is the last attempt, create fallback
                    if attempt == max_code_attempts:
                        print(f"❌ All {max_code_attempts} code generation attempts failed")
                        visualization_data = {
                            "type": "text_analysis",
                            "title": "Data Analysis",
                            "description": f"Unable to generate visualization after {max_code_attempts} attempts",
                            "chart_type": "text",
                            "error_info": f"Code execution failed after {max_code_attempts} attempts"
                        }
        else:
            print("ℹ️ No data available for visualization")
        
        return {
            **state,
            "visualization_data": visualization_data
        }
    
    def _generate_and_execute_visualization_code(self, user_question: str, data_summary: str, 
                                               attempt: int, previous_errors: List[dict]) -> dict:
        """Generate and execute visualization code with error context"""
        
        # Base system prompt for visualization
        base_prompt = """You are a data visualization expert. Create Python matplotlib code to visualize the provided data.

TASK: Based on the user's question and data, generate complete Python matplotlib code that:
1. Imports necessary libraries (matplotlib.pyplot, pandas, numpy, etc.)
2. Creates the data structure from the provided dataset
3. Generates an appropriate chart
4. Saves the image to a BytesIO buffer
5. Returns the image as base64 string

RETURN FORMAT: Respond with ONLY executable Python code (no markdown, no explanations):

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from matplotlib import colors

# Set text color to black for readability
plt.rcParams['text.color'] = 'black'

# Create the data
data = {your_data_structure_here}
df = pd.DataFrame(data)

# Create the visualization
plt.figure(figsize=(12, 8))
# Your chart code here...
plt.title('Your Title', color='black', fontsize=16)
plt.xlabel('X Label', color='black', fontsize=12)
plt.ylabel('Y Label', color='black', fontsize=12)
plt.gca().tick_params(colors='black', labelsize=10)
plt.xticks(rotation=45, color='black')
plt.grid(True, alpha=0.3, color='gray')
plt.tight_layout()

# Save to BytesIO and convert to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.read()).decode()
plt.close()

# Return the result
result = {
    "type": "image",
    "image_data": f"data:image/png;base64,{image_base64}",
    "title": "Your Chart Title",
    "description": "Description of insights and why this visualization is effective",
    "chart_type": "bar"  # or pie, line, scatter, etc.
}
```

IMPORTANT GUIDELINES:
- Use colorful, professional color schemes with HIGH CONTRAST
- CRITICAL TEXT READABILITY: Always use BLACK text on white/light backgrounds
- Set plt.rcParams['text.color'] = 'black' at the beginning
- Use plt.gca().tick_params(colors='black', labelsize=10)
- Set proper title and label colors with black text
- Make all text clearly visible with proper font sizes
- Choose appropriate chart types for the data
- Handle long category names appropriately (rotation, truncation)
- For ranking data: use bar/horizontal bar charts
- For percentages/parts of whole: use pie charts
- For trends: use line charts
- For correlations: use scatter plots
- Always add: plt.grid(True, alpha=0.3, color='gray') for better readability
- CRITICAL: Always include plt.close() to free memory

PIE CHART SPECIFIC REQUIREMENTS:
- NEVER use shadow=True in pie charts (causes visual artifacts)
- Always use shadow=False explicitly
- Use clear, distinct colors from matplotlib.colors or custom color palettes
- Set explode parameter to (0.05, 0, 0, 0, 0...) to slightly separate slices for clarity
- Use autopct='%1.1f%%' for percentage display
- Set startangle=90 for better visual alignment
- Use plt.axis('equal') to ensure perfect circle shape
- Example pie chart format:
```python
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
explode = (0.05, 0, 0, 0, 0)[:len(data)]
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
         colors=colors[:len(data)], explode=explode, shadow=False, 
         textprops={'color': 'black', 'fontsize': 10})
plt.axis('equal')
```

DATA CONTEXT: NYC 311 Service Request Data - Focus on clear, actionable insights

CHART-SPECIFIC TEMPLATES:

🥧 PIE CHART TEMPLATE (for categorical distributions):
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64

# Set text color to black for readability
plt.rcParams['text.color'] = 'black'

# Create data (example)
data = {'Category A': 25, 'Category B': 30, 'Category C': 20, 'Category D': 15, 'Category E': 10}
df = pd.DataFrame(list(data.items()), columns=['Category', 'Count'])

# Define colors and explode (NO SHADOWS)
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#a8e6cf', '#dcedc1']
explode = [0.05 if i == 0 else 0 for i in range(len(df))]  # Explode largest slice

plt.figure(figsize=(10, 8))
plt.pie(df['Count'], labels=df['Category'], autopct='%1.1f%%', startangle=90,
         colors=colors[:len(df)], explode=explode, shadow=False,
         textprops={'color': 'black', 'fontsize': 11, 'weight': 'bold'})
plt.title('Distribution Title', color='black', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures circular pie

# Save and return
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.read()).decode()
plt.close()

result = {
    "type": "image",
    "image_data": f"data:image/png;base64,{image_base64}",
    "title": "Distribution Chart",
    "description": "Clear pie chart showing proportional distributions",
    "chart_type": "pie"
}
```

BAR CHART TEMPLATE (for rankings/comparisons):
```python
# Similar structure but with plt.bar() instead of plt.pie()
# Use plt.xticks(rotation=45) for long labels
# Add plt.grid(True, alpha=0.3) for better readability
```"""

        # Add error context for retry attempts
        if attempt > 1 and previous_errors:
            error_context = "\n\nPREVIOUS EXECUTION ERRORS (fix these issues):\n"
            for i, error_info in enumerate(previous_errors[-2:], 1):  # Show last 2 errors
                error_context += f"Attempt {error_info['attempt']}: {error_info['error']}\n"
            
            error_context += """
COMMON FIXES:
- Use proper data type conversions (e.g., pd.to_numeric() for numbers)
- Handle missing/null values with .dropna() or .fillna()
- Ensure column names match exactly (case-sensitive)
- Add try/except blocks for data operations
- Use .astype(str) for categorical data if needed
- Check data structure before plotting
- Use proper matplotlib color specifications

🥧 PIE CHART SPECIFIC FIXES:
- NEVER use shadow=True (causes broken/distorted charts)
- Always set shadow=False explicitly
- Use plt.axis('equal') to prevent oval shapes
- Ensure explode list length matches data length: explode=explode[:len(data)]
- Use proper color specifications: colors=colors[:len(data)]
- Set white background: plt.savefig(..., facecolor='white')
- Use clear text properties: textprops={'color': 'black', 'fontsize': 11, 'weight': 'bold'}
- For small slices, consider combining into "Others" category if < 2%
"""
            base_prompt += error_context
        
        messages = [
            SystemMessage(content=base_prompt),
            HumanMessage(content=f"""
User Question: {user_question}

{data_summary}

Please analyze this data and determine the best visualization approach.
{"" if attempt == 1 else f"This is retry attempt #{attempt}. Please fix the previous errors mentioned above."}
""")
        ]
        
        # Generate code with LLM
        response = self.client.invoke(messages, timeout=120, max_tokens=3000)
        viz_response = response.content
        
        # Extract and execute Python code
        python_code = viz_response
        if "```python" in python_code:
            python_code = python_code.split("```python")[1].split("```")[0].strip()
        elif "```" in python_code:
            python_code = python_code.split("```")[1].split("```")[0].strip()
        
        print(f"🐍 Executing Python visualization code (attempt {attempt})...")
        
        # Create a safe execution environment
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Execute the LLM-generated code
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import base64
        
        exec_globals = {
            'matplotlib': matplotlib,
            'plt': plt,
            'pd': pd,
            'np': np,
            'io': io,
            'base64': base64,
            'colors': matplotlib.colors,
        }
        exec_locals = {}
        
        exec(python_code, exec_globals, exec_locals)
        
        # Get the result from the executed code
        if 'result' in exec_locals:
            visualization_data = exec_locals['result']
            print(f"✅ Generated {visualization_data.get('chart_type', 'unknown')} visualization")
            return visualization_data
        else:
            raise Exception("No 'result' variable found in executed code")
        
    def _generate_final_answer(self, state: SQLAgentState) -> SQLAgentState:
        """Generate final natural language answer"""
        print("📝 Generating final answer...")
        
        if state["query_result"] is not None and not state["query_result"].empty:
            # Format query result for LLM with enhanced statistical context
            result_str = state["query_result"].to_string(index=False, max_rows=25)
            
            # Add statistical summary information
            result_summary = f"""
QUERY RESULT SUMMARY:
- Total rows returned: {len(state["query_result"])}
- Columns in result: {list(state["query_result"].columns)}
"""
            
            # Add numerical summaries for numeric columns
            numeric_cols = state["query_result"].select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                result_summary += "- Numerical statistics:\n"
                for col in numeric_cols:
                    col_stats = state["query_result"][col].describe()
                    result_summary += f"  * {col}: Total={col_stats.get('sum', 'N/A')}, Avg={col_stats.get('mean', 'N/A'):.2f}, Range={col_stats.get('min', 'N/A')}-{col_stats.get('max', 'N/A')}\n"
            
            system_prompt = """You are a senior data analyst specializing in NYC 311 service request analytics. Provide comprehensive, insight-rich responses that maximize the value of numerical data, formatted in clean HTML for optimal display.

RESPONSE FORMAT REQUIREMENTS:
- Format your entire response as valid HTML that will be displayed directly in a web interface
- Use proper HTML tags (<h1>, <h2>, <p>, <ul>, <li>, <table>, etc.) for structured presentation
- Do NOT use markdown syntax - use HTML exclusively
- Include CSS classes that follow this convention: 'data-insight', 'data-summary', 'data-highlight', 'data-table'

HTML STRUCTURE TEMPLATE:
```
<div class="data-response">
  <h1 class="data-title">Main Answer to the Question</h1>
  
  <div class="data-summary">
    <p>Brief executive summary with key findings</p>
  </div>
  
  <div class="data-insight">
    <h2>Key Insights</h2>
    <ul>
      <li><strong>Important Finding:</strong> Details with numbers and percentages</li>
      <!-- Additional insights -->
    </ul>
  </div>
  
  <div class="data-details">
    <h2>Detailed Analysis</h2>
    <!-- Analysis sections with numerical details -->
  </div>
  
  <div class="data-highlight">
    <h3>Most Significant Patterns</h3>
    <!-- Highlight important patterns -->
  </div>
  
  <div class="data-conclusion">
    <h2>Conclusion & Recommendations</h2>
    <p>Summary with actionable insights</p>
  </div>
</div>
```

CONTENT EXCELLENCE GUIDELINES:

📊 NUMERICAL EMPHASIS:
- Lead with the most important numbers (totals, percentages, key metrics)
- Include ALL relevant statistics from the query results
- Add context by comparing numbers (ratios, differences, trends)
- Use specific figures rather than vague terms ("23,456 requests" not "many requests")
- Calculate and mention percentages, averages, and relative proportions
- Use <strong> tags to highlight important numbers

🔍 ANALYTICAL DEPTH:
- Identify patterns, trends, and anomalies in the data
- Provide comparative analysis (highest vs lowest, before vs after)
- Highlight significant findings and outliers
- Mention data quality observations if relevant
- Present complex data in proper HTML tables using <table>, <tr>, <th>, and <td> tags

💡 BUSINESS INSIGHTS:
- Explain what the numbers mean in practical terms
- Suggest implications for city services, resource allocation, or citizen satisfaction
- Identify actionable insights from the data patterns
- Connect findings to real-world NYC service delivery
- Include a "Recommendations" section when appropriate

📝 VISUAL STRUCTURE:
- Use color-indicating classes: "text-success" for positive trends, "text-danger" for concerning findings
- Create proper hierarchy with h1, h2, h3 heading levels
- Use bullet points (<ul> and <li>) for multiple data points
- For important metrics, use <div class="metric-card"><h3>Label</h3><p class="value">Value</p></div>
- Format percentages and statistics consistently with proper rounding

� ABSOLUTE CRITICAL: USE ONLY THE PROVIDED CSS - NO MODIFICATIONS:
- Use the EXACT CSS template provided above with !important declarations
- DO NOT add any additional CSS properties or modify existing ones
- DO NOT add color, background-color, or any other styling
- The !important declarations ensure white background and black text always
- This prevents any invisible text issues completely
- FORBIDDEN: Adding any CSS beyond the exact template provided
- FORBIDDEN: Modifying colors, backgrounds, or any styling properties
- The provided CSS forces maximum readability with white backgrounds and black text

�🎯 PRECISION REQUIREMENTS:
- Include exact counts, percentages (to 1-2 decimal places), and statistical measures
- Mention data timeframes and scope when relevant
- Reference specific categories, locations, or agencies by name
- Acknowledge any limitations or data gaps

EXAMPLE ENHANCED RESPONSE STYLE:
Instead of plain text: "Brooklyn has the most complaints"

Use HTML format:
```html
<div class="data-insight">
  <p><strong>Brooklyn</strong> leads with <span class="highlight">89,247</span> service requests (<strong>24.8%</strong> of total), followed by Queens with 76,123 requests (21.2%). This represents a significant <span class="text-danger">17% higher volume</span> than Queens, indicating either higher population density or more active civic engagement in Brooklyn.</p>
</div>
```

CRITICAL: Use this EXACT CSS with forced readable colors - DO NOT MODIFY:
<style>
.data-response { font-family: Arial, sans-serif; line-height: 1.6; background: white !important; color: black !important; }
.data-title { border-bottom: 2px solid gray; padding-bottom: 10px; margin-bottom: 20px; background: white !important; color: black !important; }
.data-summary { padding: 15px; margin: 15px 0; border: 1px solid gray; background: white !important; color: black !important; }
.metric-cards { display: flex; gap: 15px; margin: 20px 0; }
.metric-card { flex: 1; padding: 15px; border: 1px solid gray; text-align: center; background: white !important; color: black !important; }
.metric-card h3 { font-size: 14px; margin: 0 0 10px 0; background: white !important; color: black !important; }
.metric-card .value { font-size: 24px; font-weight: bold; margin: 0; background: white !important; color: black !important; }
.data-insight { padding: 15px; margin: 15px 0; border: 1px solid gray; background: white !important; color: black !important; }
.data-highlight { padding: 15px; margin: 15px 0; border: 1px solid gray; background: white !important; color: black !important; }
.data-conclusion { padding: 15px; margin: 15px 0; border: 1px solid gray; background: white !important; color: black !important; }
.data-table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white !important; }
.data-table th { border: 1px solid gray; padding: 10px; text-align: left; font-weight: bold; background: white !important; color: black !important; }
.data-table td { border: 1px solid gray; padding: 10px; background: white !important; color: black !important; }
.highlight-item { margin-bottom: 15px; background: white !important; color: black !important; }
strong { font-weight: bold; background: white !important; color: black !important; }
</style>
IMPORTANT: Use this EXACT CSS - do not add, remove, or modify ANY properties. The !important ensures text is always visible.

REMEMBER: Your entire response must be valid HTML that can be directly inserted into a web page without further processing.
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
User Question: {state['user_question']}

SQL Query Used: {state['sql_query']}

{result_summary}

Detailed Query Results:
{result_str}

ANALYSIS REQUIREMENTS:
1. Provide a comprehensive answer with ALL relevant numbers from the results
2. Include percentages, totals, averages, and comparative statistics
3. Highlight the most significant findings and patterns
4. Explain what these numbers mean for NYC services and citizens
5. Structure the response for maximum clarity and impact

Please provide a detailed, insight-rich natural language answer based on these results.
""")
            ]
            
            # Use longer timeout and more tokens for final answer generation
            response = self.client.invoke(messages, timeout=180, max_tokens=6000, retry_count=3)
            final_answer = response.content
            
        elif state["query_result"] is not None and state["query_result"].empty:
            # Query executed successfully but returned no data
            final_answer = "No data found matching your query criteria in the database."
        else:
            final_answer = f"I apologize, but I couldn't successfully generate a working SQL query to answer your question after {state['attempt_count']} attempts. The last error was: {state.get('error_message', 'Unknown error')}"
        
        print("✅ Final answer generated")
        
        return {
            **state,
            "final_answer": final_answer
        }
    
    def ask_question(self, question: str, conversation_history: List[dict] = None) -> dict:
        """
        Ask a question and get an answer using the LangGraph workflow
        
        Args:
            question: User's question about the data
            conversation_history: Optional list of previous messages
            
        Returns:
            dict: Complete state with final answer
        """
        print(f"\n🤖 Processing question: {question}")
        
        if conversation_history is None:
            conversation_history = []
        
        # Initialize state
        initial_state = {
            "user_question": question,
            "conversation_history": conversation_history,
            "table_schema": self._get_schema_string(),
            "sql_query": "",
            "query_result": None,
            "error_message": None,
            "attempt_count": 0,
            "max_attempts": self.max_attempts,
            "final_answer": "",
            "requires_clarification": False,
            "clarification_question": "",
            "visualization_data": None,
            "messages": [],
            "result_type": None,
            "previous_code_errors": []
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def close(self):
        """Close database connections"""
        # Since we're now using per-query connections, 
        # there's no global connection to close
        pass
    
    def get_response(self, message: str, conversation_history: List[dict] = None) -> dict:
        """
        Get response from the SQL Agent for a user message
        This method provides a simple interface for Flask integration
        
        Args:
            message: User's question/message
            conversation_history: Optional list of previous messages
            
        Returns:
            dict: Response object containing the answer and clarification status
        """
        try:
            result = self.ask_question(message, conversation_history)
            return {
                "response": result.get('final_answer', 'Sorry, I could not process your question.'),
                "requires_clarification": result.get('requires_clarification', False),
                "clarification_question": result.get('clarification_question', ''),
                "visualization": result.get('visualization_data', None)
            }
        except Exception as e:
            error_message = f"Error processing your question: {str(e)}"
            print(f"❌ {error_message}")
            return {
                "response": error_message,
                "requires_clarification": False,
                "visualization": None
            }
