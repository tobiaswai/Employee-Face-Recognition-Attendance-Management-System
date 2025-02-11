from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.
	

class Present(models.Model):
    STATUS_CHOICES = [
        ('P', 'Present'),
        ('A', 'Absent'),
        ('L', 'Late'),
        ('E', 'Excused'),
    ]
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    date = models.DateField(default=datetime.date.today)
    present=models.BooleanField(default=False)
    status = models.CharField(max_length=1, choices=STATUS_CHOICES, default='A')
class Time(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	time=models.DateTimeField(null=True,blank=True)
	out=models.BooleanField(default=False)
class Shift(models.Model):
    SHIFT_CHOICES = (
        ('A', 'Shift A (04:00 - 12:00)'),
        ('B', 'Shift B (12:00 - 20:00)'),
        ('C', 'Shift C (20:00 - 04:00)'), 
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, unique=True)
    shift_type = models.CharField(max_length=1, choices=SHIFT_CHOICES)
    start_time = models.TimeField(editable=False)
    end_time = models.TimeField(editable=False)

    def save(self, *args, **kwargs):
        # Define start and end times based on shift type
        shift_times = {
            'A': ('20:00:00', '04:00:00'),
            'B': ('04:00:00', '12:00:00'),
            'C': ('12:00:00', '20:00:00'),
        }
        times = shift_times.get(self.shift_type, ('00:00:00', '00:00:00'))
        self.start_time, self.end_time = times
        super().save(*args, **kwargs)
        
class Events(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255,null=True,blank=True)
    start = models.DateTimeField(null=True,blank=True)
    end = models.DateTimeField(null=True,blank=True)
    class Meta:  
        db_table = "tblevents"