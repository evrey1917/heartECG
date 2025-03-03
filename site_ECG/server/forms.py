from django import forms
from .models import PatientData

class PatientForm(forms.ModelForm):
    class Meta:
        model = PatientData
        fields = ['ekg_file', 'height', 'weight']