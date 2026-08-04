

def calculate_future_value(
    n_years,
    initial_amount,
    finalpredict
):
    # ตรวจสอบว่าค่า n_years
    if n_years < 1: n_years = 1

    # คำนวณเงินหลังจากการบวกเงินเฟ้อในแต่ละปี
    amount = initial_amount
    result = []  # เพื่อเก็บจำนวนเงินของแต่ละปี

    for i in range(n_years):
        inflat_rate = finalpredict[i] / 100  # เปลี่ยนจาก % เป็นทศนิยม
        amount += amount*inflat_rate # บวกเปอร์เซ็นต์เงินเฟ้อในปีนั้น
        result.append({
            'year': i+1,
            'inflation_rate': finalpredict[i],
            'amount': round(amount,2) # ปัดเศษให้เหลือ 2 ตำแหน่ง
        })
    
    return result