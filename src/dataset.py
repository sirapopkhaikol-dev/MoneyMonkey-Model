import pandas as pd

def load_data_from_google_sheet():

    sheet_id = "1_NMyB8X2HOFj09tU6rtjmRwEuT6E1fK1El9E-ceA0Pw"
    sheet_name = "Sheet1"   # เปลี่ยนเป็นชื่อ Sheet ที่ต้องการ ถ้า sheet ชื่ออื่นให้ใส่ตรงนี้
    
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)

    return df