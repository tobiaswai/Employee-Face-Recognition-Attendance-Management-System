from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.urls import reverse
import datetime

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
        ('D', 'Day Off'),
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, unique=True)
    shift_type = models.CharField(max_length=1, choices=SHIFT_CHOICES)
    start_time = models.TimeField(editable=False)
    end_time = models.TimeField(editable=False)

    def save(self, *args, **kwargs):
        # Define start and end times based on shift type
        shift_times = {
            'A': ('21:00:00', '05:00:00'),
            'B': ('05:00:00', '13:00:00'),
            'C': ('13:00:00', '21:00:00'),
        }
        times = shift_times.get(self.shift_type, ('00:00:00', '00:00:00'))
        self.start_time, self.end_time = times
        super().save(*args, **kwargs)
        
class Event(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()

    @property
    def get_html_url(self):
        url = reverse('cal:event_edit', args=(self.id,))
        return f'<a href="{url}"> {self.title} </a>'
    
class ShiftCalendar(models.Model):
    users = models.ManyToManyField(User)  # Changed from ForeignKey to ManyToManyField
    date = models.DateField()
    shift_type = models.CharField(max_length=1, choices=[
        ('A', 'Shift A (04:00 - 12:00)'),
        ('B', 'Shift B (12:00 - 20:00)'),
        ('C', 'Shift C (20:00 - 04:00)'),
        ('D', 'Day Off')
    ])

    def __str__(self):
        user_names = ', '.join([user.username for user in self.users.all()])
        return f"{user_names} - {self.date} - {self.get_shift_type_display()}"