from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self, output_folder='reports'):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
    def generate_report(self, filename, has_crack, confidence, 
                       severity_result=None, original_img_path=None, 
                       heatmap_img_path=None, bbox_img_path=None):
        """Generate PDF report for crack detection"""
        
        # Create PDF filename
        pdf_filename = os.path.join(
            self.output_folder, 
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        # Create document
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("Infrastructure Crack Detection Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        metadata = [
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Image File:', filename],
            ['Analysis Status:', 'Complete']
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detection Result
        result_title = Paragraph("<b>Detection Result</b>", self.styles['Heading2'])
        story.append(result_title)
        story.append(Spacer(1, 0.1*inch))
        
        if has_crack:
            result_text = f"<font color='red'><b>CRACK DETECTED</b></font>"
            result_confidence = f"Confidence: {confidence*100:.2f}%"
        else:
            result_text = f"<font color='green'><b>NO CRACK DETECTED</b></font>"
            result_confidence = f"Confidence: {confidence*100:.2f}%"
        
        story.append(Paragraph(result_text, self.styles['Normal']))
        story.append(Paragraph(result_confidence, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Severity Analysis (if applicable)
        if has_crack and severity_result:
            severity_title = Paragraph("<b>Severity Analysis</b>", self.styles['Heading2'])
            story.append(severity_title)
            story.append(Spacer(1, 0.1*inch))
            
            severity_data = [
                ['Severity Level:', f"{severity_result['severity_level']}"],
                ['Severity Score:', f"{severity_result['severity_score']:.1f}/100"],
                ['Affected Area:', f"{severity_result['area_percentage']:.2f}%"],
                ['Estimated Length:', f"{severity_result['estimated_length']:.1f} pixels"],
                ['Estimated Width:', f"{severity_result['estimated_width']:.1f} pixels"],
                ['Recommendation:', severity_result['recommendation']]
            ]
            
            severity_table = Table(severity_data, colWidths=[2*inch, 4*inch])
            severity_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
            ]))
            story.append(severity_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Images
        if original_img_path or heatmap_img_path or bbox_img_path:
            images_title = Paragraph("<b>Visual Analysis</b>", self.styles['Heading2'])
            story.append(images_title)
            story.append(Spacer(1, 0.1*inch))
            
            img_width = 2*inch
            img_height = 2*inch
            
            if original_img_path and os.path.exists(original_img_path):
                story.append(Paragraph("<b>Original Image:</b>", self.styles['Normal']))
                img = Image(original_img_path, width=img_width, height=img_height)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            
            if has_crack:
                if heatmap_img_path and os.path.exists(heatmap_img_path):
                    story.append(Paragraph("<b>Crack Heatmap:</b>", self.styles['Normal']))
                    img = Image(heatmap_img_path, width=img_width, height=img_height)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                
                if bbox_img_path and os.path.exists(bbox_img_path):
                    story.append(Paragraph("<b>Detected Crack Location:</b>", self.styles['Normal']))
                    img = Image(bbox_img_path, width=img_width, height=img_height)
                    story.append(img)
        
        # Disclaimer
        story.append(Spacer(1, 0.5*inch))
        disclaimer = Paragraph(
            "<i><font size=8>Disclaimer: This report is generated by an AI-powered crack detection system "
            "and should be used for preliminary assessment only. Professional structural engineering "
            "inspection is recommended for critical infrastructure decisions.</font></i>",
            self.styles['Normal']
        )
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        
        return pdf_filename