"""attendance_system_facial_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from recognition import views as recog_views
from employee import views as employee_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', recog_views.home, name='home'),
    
    path('dashboard/', recog_views.dashboard, name='dashboard'),
    path('train/', recog_views.train, name='train'),
    path('add_photos/', recog_views.add_photos, name='add-photos'),
    path('employees/', recog_views.employee_list, name='employee_list'),
    path('employees/<int:id>/edit/', recog_views.employee_edit, name='employee_edit'),
    path('employees/<int:id>/delete/', recog_views.employee_delete, name='employee_delete'),
    
    path('login/',auth_views.LoginView.as_view(template_name='employee/login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='recognition/home.html'),name='logout'),
    path('register/', employee_views.register, name='register'),
    path('mark_your_attendance', recog_views.mark_your_attendance ,name='mark-your-attendance'),
    path('mark_your_attendance_out', recog_views.mark_your_attendance_out ,name='mark-your-attendance-out'),
    path('view_attendance_home', recog_views.view_attendance_home ,name='view-attendance-home'),
       
    path('view_attendance_date', recog_views.view_attendance_date ,name='view-attendance-date'),
    path('view_attendance_employee', recog_views.view_attendance_employee ,name='view-attendance-employee'),
    path('view_my_attendance', recog_views.view_my_attendance_employee_login ,name='view-my-attendance-employee-login'),
    path('not_authorised', recog_views.not_authorised, name='not-authorised'),
     
    path('calender', recog_views.calender_index, name='calender'), 
    path('all_events/', recog_views.calender_all_events, name='calender_all_events'), 
    path('add_event/', recog_views.calender_add_event, name='calender_add_event'), 
    path('update/', recog_views.calender_update, name='calender_update'),
    path('remove/', recog_views.calender_remove, name='calender_remove'),
    
    path('calendar/', recog_views.calendar_view, name='calendar'),
    path('api/shifts/', recog_views.shifts_api, name='shifts_api'),
    path('add_shift/', recog_views.add_shift, name='add_shift'),
    path('edit_shift/<int:shift_id>/', recog_views.edit_shift, name='edit_shift'),
]
