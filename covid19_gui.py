from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your model here
model = tf.keras.models.load_model('Covid_19_Detection.h5')

def prediction(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    results = model.predict(img)
    return results

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/photos')
def photos():
    return render_template('photos.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Assuming you have a form field named 'file'
        uploaded_file = request.files['file']

        if uploaded_file:
            img_path = f"uploads/{uploaded_file.filename}"
            uploaded_file.save(img_path)

            results = prediction(img_path)

            if results[0][0] == 0:
                result_text = 'Positive For Covid-19'
            else:
                result_text = 'Negative For Covid-19'

            return render_template('result.html', image_path=img_path, result=result_text)

    return render_template('upload.html')



@app.route('/download', methods=['POST'])
def download_report():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    phone_number = request.form.get('phone_number')
    address = request.form.get('address')
    result = request.form.get('result')
    date = request.form.get('date')
    details=request.form.get('details')

    # Generate PDF
    pdf_path = generate_pdf(first_name, last_name, phone_number, address, result, date ,details)

    # Return PDF for download
    return send_file(pdf_path, as_attachment=True)

def generate_pdf(first_name, last_name, phone_number, address, result, date ,details):
    pdf_path = f'report_{first_name}_{last_name}.pdf'

    # Create a PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.blue,
        spaceAfter=12
    )
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        spaceAfter=12
    )

    # Build the content
    content = []
    content.append(Paragraph('COVID-19 Test Report', title_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Name:           {first_name} {last_name}", normal_style))
    content.append(Paragraph(f"Phone Number:     {phone_number}", normal_style))
    content.append(Paragraph(f"Address:        {address}", normal_style))
    content.append(Paragraph(f"<h1>Result:       {result}</h1>", normal_style))
    content.append(Paragraph(f"Date:     {date}", normal_style))
    content.append(Paragraph(f"Details:     {details}", normal_style))


    # Add content to the PDF document
    doc.build(content)

    return pdf_path
if __name__ == '__main__':
    app.run(debug=True)
