from django.forms import ModelForm
from django.contrib.auth.models import User
from django import forms
from employee.models import Shift, ShiftCalendar

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
        
class ShiftForm(forms.ModelForm):
    class Meta:
        model = ShiftCalendar
        fields = ['users', 'date', 'shift_type']

    def __init__(self, *args, **kwargs):
        super(ShiftForm, self).__init__(*args, **kwargs)
        self.fields['users'].widget = forms.CheckboxSelectMultiple()
        self.fields['users'].queryset = User.objects.all()
        self.fields['date'].widget = forms.DateInput(attrs={'type': 'date'})