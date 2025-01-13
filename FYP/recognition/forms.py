from django.forms import ModelForm
from django.contrib.auth.models import User
from django import forms
from employee.models import Shift

#from django.contrib.admin.widgets import AdminDateWidget

class usernameForm(forms.Form):
	username=forms.CharField(max_length=30)

class DateForm(forms.Form):
	date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))

class UsernameAndDateForm(forms.Form):
	username=forms.CharField(max_length=30)
	date_from = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
	date_to = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
 
class DateForm_2(forms.Form):
	date_from=forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
	date_to=forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
class ShiftEdit(forms.ModelForm):
    class Meta:
        model = Shift
        fields = ['shift_type']
        
class EmployeeForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email']