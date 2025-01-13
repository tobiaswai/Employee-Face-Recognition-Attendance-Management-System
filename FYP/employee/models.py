from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.
	

class Present(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	present=models.BooleanField(default=False)
	
class Time(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	time=models.DateTimeField(null=True,blank=True)
	out=models.BooleanField(default=False)
class Shift(models.Model):
    SHIFT_CHOICES = (
        ('A', 'Shift A (08:00 - 16:00)'),
        ('B', 'Shift B (16:00 - 00:00)'),
        ('C', 'Shift C (00:00 - 08:00)'), 
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, unique=True)
    shift_type = models.CharField(max_length=1, choices=SHIFT_CHOICES)
    start_time = models.TimeField(editable=False)
    end_time = models.TimeField(editable=False)

    def save(self, *args, **kwargs):
        # Define start and end times based on shift type
        shift_times = {
            'A': ('08:00', '16:00'),
            'B': ('16:00', '00:00'),
            'C': ('00:00', '08:00'),
        }
        times = shift_times.get(self.shift_type, ('00:00', '00:00'))
        self.start_time, self.end_time = times
        super().save(*args, **kwargs)