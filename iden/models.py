# models.py (inside image_classification app)
from django.db import models

class Prediction(models.Model):
    image = models.ImageField(upload_to='predictions/')
    result = models.IntegerField()
    class_namelocal = models.CharField(max_length=100)
    class_scientific_name = models.CharField(max_length=100)
    class_properties = models.TextField()
    class_use = models.TextField()

    def __str__(self):
        return f'Prediction for {self.image}'

class TrainInformation(models.Model):
    image = models.ImageField(upload_to='trainings/')
    accuracy = models.FloatField()

    def __str__(self):
        return f'Training Image {self.pk} - Accuracy: {self.accuracy}'
