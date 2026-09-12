# 🏥 SmartPharm — Pharmacy Inventory & Demand Intelligence

A Streamlit app that helps a pharmacy manage stock, track sales, forecast
demand, and catch expiring medicine before it becomes a loss.

Built to show how raw pharmacy data (sales, inventory, medicine details) can
be turned into an easy-to-use tool for real decisions — what to reorder,
what's selling well, and what needs to be cleared before it expires.

---

## 🔍 What it does

**📦 Inventory**
See every medicine and every batch in stock, with quantity, price, and
expiry date. Filter by medicine to drill into batch-level detail.

**📈 Sales Analytics**
Revenue, profit, and margin trends over time. See which medicines sell the
most, which are most profitable, and how price relates to volume.

**🤖 Demand Forecast**
Predicts how much of a medicine will sell over the next 7, 14, or 30 days,
using a machine learning model (Random Forest) trained on past sales
patterns.

**⏰ Expiry Tracking**
Flags every batch as Safe, Warning, Critical, or Expired based on days left
before expiry, and shows how much stock value is at risk. The thresholds
for "Critical" and "Warning" are adjustable — a manager can set them to
match how fast their pharmacy can actually sell or return stock.

---

## 🛠 Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** | Main programming language |
| **Pandas** | Data cleaning and analysis |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Machine learning |
| **Random Forest** | Demand forecasting |
| **Streamlit** | Web application |
| **Plotly** | Data visualization |

---

## 📁 Data files needed

**Sales CSV** — needs: `sale_date`, `medicine`, `quantity_sold`

**Inventory CSV** — needs: `inventory_id`, `medicine_id`, `batch_number`,
`expiry_date`, `current_stock`, `minimum_stock_level`, `reorder_level`

**Medicines CSV** — needs: `medicine_id`, `medicine`, `category`, `company`,
`purchase_price`, `selling_price`, `supplier`

Column names don't have to match exactly — common variations are recognized
automatically (e.g. `qty` → `quantity_sold`, `mrp` → `selling_price`).

---
Installation

Follow these steps to run the SmartPharm – Inventory & Demand Intelligence project on your computer.

1. Clone the Repository

Open Command Prompt or Anaconda Prompt and run:

```bash
git clone https://github.com/your-username/SmartPharm.git
```
```bash
cd SmartPharm
```

2. Install Required Libraries

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

The project requires:

streamlit
pandas
numpy
plotly
scikit-learn

3. Run the Application

Start the Streamlit application using:
```bash
streamlit run app.py
```

Replace app.py with the actual Python filename if your file has a different name.

4. Upload the Dataset Files

After the application opens in your browser, upload these three CSV files through the sidebar:

### Dataset Files

- 📄 **Sales CSV**
- 📦 **Inventory CSV**
- 💊 **Medicine CSV**

The application will validate and process the uploaded files automatically.

---

## 👤 About me

Built by **Prasoon Pranjal**, a B.Pharm student exploring **data analysis and data-driven problem solving**.

