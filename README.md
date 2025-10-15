## Getting Started

You'll need Python 3.8+, Node.js 16+, and a DeepSeek API key.

### Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Setup the database:**
   
   Download the NYC 311 Service Requests dataset and set up the database:
   
   a. Download the CSV file:
   - Go to [NYC Open Data - 311 Service Requests](https://www.kaggle.com/datasets/new-york-city/ny-311-service-requests)
   - Download the CSV file as `311_Service_Requests_from_2010_to_Present.csv`
   - Place it in the `backend/data/` folder
   
   b. Create the SQLite database:
   ```bash
   cd backend/data
   python csv_to_sql.py
   ```
   
   This will create a `database.db` file in the `backend/data/` folder with all the 311 service request data.

3. **Add your API key:**
   
   Create a `.env` file in the backend folder:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

3. **Install frontend:**
   ```bash
   cd frontend
   npm install
   ```

### Running the App

Start both servers in separate terminals:

**Backend:** (runs on port 8000)
```bash
cd backend
python app.py
```

**Frontend:** (runs on port 5173)
```bash
cd frontend  
npm run dev
```

Then open http://localhost:5173 in your browser.


**Backend**
- Python + Flask for the API
- SQLite database with NYC 311 data (created from CSV during setup)
- LangGraph for managing the query workflow
- DeepSeek API for natural language processing

**Frontend**
- React app with Vite
- Chart.js for data visualization