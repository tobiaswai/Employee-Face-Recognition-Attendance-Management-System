from django.contrib import admin
from .models import Time,Present,Shift, ShiftCalendar

# Register your models here.
admin.site.register(Time)
admin.site.register(Present)
admin.site.register(Shift)
admin.site.register(ShiftCalendar)