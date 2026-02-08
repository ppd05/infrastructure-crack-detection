import cv2
import numpy as np

class CrackSeverityAnalyzer:
    def __init__(self):
        self.severity_levels = {
            'LOW': (0, 30),
            'MEDIUM': (30, 60),
            'HIGH': (60, 100)
        }
    
    def calculate_crack_area_percentage(self, heatmap, threshold=150):
        """Calculate percentage of image containing cracks"""
        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        total_pixels = binary.shape[0] * binary.shape[1]
        crack_pixels = np.sum(binary > 0)
        
        percentage = (crack_pixels / total_pixels) * 100
        return percentage
    
    def calculate_crack_length(self, bbox_list):
        """Estimate total crack length from bounding boxes"""
        total_length = 0
        for (x, y, w, h) in bbox_list:
            # Approximate length as diagonal
            length = np.sqrt(w**2 + h**2)
            total_length += length
        return total_length
    
    def calculate_crack_width(self, heatmap):
        """Estimate average crack width"""
        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # Calculate average width
        widths = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                _, _, w, _ = cv2.boundingRect(contour)
                widths.append(w)
        
        return np.mean(widths) if widths else 0
    
    def assess_severity(self, heatmap, bbox_list):
        """Assess overall crack severity"""
        area_pct = self.calculate_crack_area_percentage(heatmap)
        length = self.calculate_crack_length(bbox_list)
        width = self.calculate_crack_width(heatmap)
        
        # Composite severity score
        severity_score = (area_pct * 0.5) + (min(length, 200)/200 * 30) + (min(width, 50)/50 * 20)
        
        # Determine severity level
        if severity_score < self.severity_levels['LOW'][1]:
            level = 'LOW'
            color = '🟢'
            recommendation = "Monitor regularly. Minor cosmetic issue."
        elif severity_score < self.severity_levels['MEDIUM'][1]:
            level = 'MEDIUM'
            color = '🟡'
            recommendation = "Schedule inspection. Potential structural concern."
        else:
            level = 'HIGH'
            color = '🔴'
            recommendation = "Immediate action required. Significant structural risk."
        
        return {
            'severity_level': level,
            'severity_score': severity_score,
            'area_percentage': area_pct,
            'estimated_length': length,
            'estimated_width': width,
            'color_code': color,
            'recommendation': recommendation
        }