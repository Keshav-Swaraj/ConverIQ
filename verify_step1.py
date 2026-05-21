import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from core.document import load_pdf

def generate_simple_pdf(path):
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(100, 750, "This is a simple textbook page.")
    c.drawString(100, 730, "It has some clean text.")
    c.showPage()
    c.save()

def generate_long_pdf(path, num_pages=55):
    c = canvas.Canvas(path, pagesize=letter)
    for i in range(1, num_pages + 1):
        c.drawString(100, 750, f"This is page {i} of the long PDF.")
        c.drawString(100, 700, "Content " * 10)
        c.drawString(100, 50, f"- {i} -")
        c.showPage()
    c.save()

def generate_image_based_pdf(path):
    c = canvas.Canvas(path, pagesize=letter)
    # Simulate scanned PDF by just adding a rectangle, no text
    c.rect(100, 100, 400, 600, fill=1)
    c.showPage()
    c.save()

def verify():
    print("--- Verifying Phase 1 Step 1 ---")
    
    # 1. Simple PDF
    generate_simple_pdf("simple.pdf")
    text = load_pdf("simple.pdf")
    assert "simple textbook page" in text
    print("[PASS] Load simple PDF - text returned")
    
    # 2. Text is clean (no random numbers)
    generate_long_pdf("long.pdf", num_pages=55)
    text = load_pdf("long.pdf")
    assert "- 5 -" not in text
    assert "- 55 -" not in text
    assert "This is page 55" in text
    print("[PASS] Text is clean - no page numbers")
    
    # 3. Long PDF
    assert "This is page 1" in text
    assert "This is page 55" in text
    print("[PASS] Long PDF - all pages extracted, no truncation")
    
    # 4. Scanned PDF
    generate_image_based_pdf("scanned.pdf")
    text = load_pdf("scanned.pdf")
    assert text == ""
    print("[PASS] Scanned PDF handled gracefully (empty string)")
    
    # 5. File not found
    try:
        load_pdf("nonexistent.pdf")
        print("[FAIL] File not found did not raise error")
    except Exception as e:
        print("[PASS] File not found raised clear error")

    print("All Step 1 checks passed!")

if __name__ == "__main__":
    verify()
