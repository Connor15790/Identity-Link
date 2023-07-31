from django.db import models

# Create your models here.
class Person(models.Model):
    Id = models.AutoField
    First_Name = models.CharField(max_length=50, default="")
    Last_Name = models.CharField(max_length=50, default="")
    Description = models.CharField(max_length=1000, default="")
    Criminal_Record = models.CharField(max_length=100, default="")
    Occupation = models.CharField(max_length=100, default="")
    Date_of_Birth = models.DateField()
    Image = models.ImageField(upload_to="face/images", default="")
    
    def __str__(self):
        return self.First_Name