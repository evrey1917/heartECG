from django.db import models

class PatientData(models.Model):
    ekg_file = models.FileField(upload_to="ekg_data/", help_text="Загрузите файл .h5 с ЭКГ")
    height = models.FloatField()
    weight = models.FloatField()
    result = models.CharField(max_length=255, blank=True, null=True)
