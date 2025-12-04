# email_agent
Email Agent
# 📧 Email Agent – Multi-Source Semi-Agentic Email Generator

`email_agent.py` is a fast, lightweight, semi-agentic email generator that:

1. Loads contact data from **CSV**, **SQLite (.db)**, and **JSON**
2. Automatically merges all sources by a chosen key (default: `email`)
3. Searches for contacts using **keywords** across selected fields
4. Asks the user **WHY** they are sending the emails (sales, follow-up, reminders, etc.)
5. Generates personalized email subject + body for every matched contact
6. Outputs everything to **output_emails.csv**

Uses the **OpenAI Python SDK v1+** or a local template when in `--dry-run` mode.

---

## 🚀 Features

- Multi-source ingest: CSV → SQLite → JSON  
- Automatic merge & column coalescing  
- Keyword-based row filtering  
- Interactive prompt: *Why are we sending these emails?*  
- OpenAI-powered email generation (or template fallback)  
- Saves results into a clean CSV  
- Simple command-line interface  
- Works with Python 3.9+

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Typical `requirements.txt`:

```
pandas
sqlalchemy
openai
python-dotenv
```

---

## 🔑 Setting Your OpenAI API Key

### **Option A — Environment Variable (recommended)**

**Windows CMD**
```cmd
setx OPENAI_API_KEY "your_key_here"
```

**PowerShell**
```powershell
setx OPENAI_API_KEY "your_key_here"
```

**macOS/Linux**
```bash
export OPENAI_API_KEY="your_key_here"
```

---

### **Option B — Hardcode (not recommended)**

Inside the script:

```python
client = OpenAI(api_key="your_key_here")
```

---

## ▶️ Usage

### Minimum working example
```bash
python email_agent.py --csv contacts.csv --keywords "software"
```

### Use multiple data sources
```bash
python email_agent.py --csv contacts.csv --db clients.db --json extra.json --keywords "marketing, offer"
```

### Dry run (no API calls)
```bash
python email_agent.py --csv contacts.csv --keywords "follow up" --dry-run
```

### Custom fields to search
```bash
python email_agent.py --csv contacts.csv --keywords "renewal" --fields "notes,company"
```

### Custom primary key
```bash
python email_agent.py --csv contacts.csv --keywords "update" --key customer_id
```

### Use a specific model
```bash
python email_agent.py --csv contacts.csv --keywords "invoice" --model gpt-4o-mini
```

---

## 📁 Output

The script generates:

```
output_emails.csv
```

The file contains:

| email | first_name | last_name | subject | body |
|-------|-------------|------------|---------|--------|

---

## 🤖 How the Agent Works

1. Load all available data sources  
2. Normalize columns & merge on primary key  
3. Prompt user:  
   **“Why are we sending these emails?”**  
4. Search rows containing your keywords  
5. Build a combined LLM prompt using:  
   - contact info  
   - keywords  
   - the user's *why*  
   - optional custom instructions  
6. Generate subject + body for each matched contact  
7. Save results to CSV

---

## 🧪 Development Tips

- Use `--dry-run` while testing to avoid API costs  
- Your CSV should contain an `email` column (or use `--key`)  
- SQLite tables must have readable names or specify via `--table`

---
## 🤝 Contributing

Pull requests welcome, especially for:

- async OpenAI batching
- embedding-based keyword search
- HTML email template support
- REST API wrapper
- front-end UI

