# ⚡ Internal Unit Commitment Optimization Web Tool

This project provides an internal Django-based web interface for performing **Unit Commitment & Economic Dispatch optimization** using Pyomo.  
Employees upload a DAM price Excel file, and the app computes the optimal operation schedule of a thermal power unit and returns a chart.

> **Designed for corporate internal networks only.**

---

## ✨ Features

- 🔐 **Internal-only access** (IP-based restriction + login)
- 👥 Optional integration with corporate authentication (LDAP/AD)
- ⬆️ Upload Excel file with hourly DAM prices
- ⚙️ Runs a Pyomo Unit Commitment model
- 📈 Returns optimization results as a chart (PNG, Matplotlib)
- 🧮 Considers realistic constraints:
  - Minimum / maximum power limits  
  - Ramp-up / ramp-down limits  
  - Startup costs and limits  
  - Coal and CO₂ costs  
  - Degradation profile  
  - Minimum annual generation requirement  

---

## 🛠️ Tech Stack

| Component | Technology |
|---------|------------|
| Backend | Django (Python) |
| Optimization | Pyomo |
| Solver | CBC or GLPK |
| Data | Pandas / NumPy |
| Charts | Matplotlib |
| Security | IP whitelist + Django auth |

---

## 📦 Installation

```bash
git clone https://github.com/your-org/uc-optimizer-web.git
cd uc-optimizer-web
pip install -r requirements.txt
