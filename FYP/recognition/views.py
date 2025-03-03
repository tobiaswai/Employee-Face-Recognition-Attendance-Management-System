from django.shortcuts import render,redirect, get_object_or_404
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2, EmployeeForm, ShiftEdit, ShiftForm
from django.utils.dateparse import parse_time
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.contrib.auth.models import User
from datetime import timedelta, date
from collections import defaultdict
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from FYP.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import _classes
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from employee.models import Present, Time, Shift, ShiftCalendar
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math

mpl.use('Agg')


#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id)

	# Detect face
	#Loading the HOG face detector and the shape predictpr for allignment

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	fa = FaceAligner(predictor , desiredFaceWidth = 300)
	#capture images from the webcam and process and detect the face
	# Initialize the video stream
	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	#time.sleep(2.0) ####CHECK######

	# Our identifier
	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
	# Our dataset naming counter
	sampleNum = 0
	# Capturing the faces one by one and detect the faces and showing it on the window
	while(True):
		# Capturing the image
		#vs.read each frame
		frame = vs.read()
		#Resize each image
		frame = imutils.resize(frame ,width = 800)
		#the returned img is a colored image but for the classifier to work we need a greyscale image
		#to convert
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#To store the faces
		#This will detect all the images in the current frame, and it will return the coordinates of the faces
		#Takes in image and some other parameter for accurate result
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
		
		
		for face in faces:
			print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			# Whenever the program captures the face, we will write that is a folder
			# Before capturing the face, we need to tell the script whose face it is
			# For that we will need an identifier, here we call it id
			# So now we captured a face, we need to write it in a file
			sampleNum = sampleNum+1
			# Saving the image dataset, but only the face part, cropping the rest
			
			if face is None:
				print("face is none")
				continue
			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)
			#cv2.imshow("Image Captured",face_aligned)
			# @params the initial point of the rectangle will be x,y and
			# @params end point will be x+width and y+height
			# @params along with color of the rectangle
			# @params thickness of the rectangle
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		cv2.waitKey(1)
		#To get out of the loop
		if(sampleNum>200):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()


def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])
	except:
		return ([-1],[0])
	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])
	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)
	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout(pad=3.0)	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()


def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
			qs=Present.objects.get(user=user,date=today)
		except:
			qs= None

		if qs is None:
			if present[person]==True:
						a=Present(user=user,date=today,present=True)
						a.save()
			else:
				a=Present(user=user,date=today,present=False)
				a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=False)
			a.save()


def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()


def check_validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)
	if(len(times_in)!=len(times_out)):
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time

		else:
			prev_time=obj.time

		prev=curr

	return (True,break_hourss)


def convert_hours_to_hours_mins(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")


#used
def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        user_shift = Shift.objects.filter(user=obj.user).first()
        if not user_shift:
            print(f"No shift assigned for user {obj.user}")
            continue
        
        times_in = time_qs.filter(date=date, out=False).order_by('time')
        times_out = time_qs.filter(date=date, out=True).order_by('time')
        times_all = time_qs.filter(date=date).order_by('time')

        obj.time_in = times_in.first().time if times_in.exists() else None
        obj.time_out = times_out.last().time if times_out.exists() else None

        if obj.time_in:
            obj.shift_start_time = user_shift.start_time
            if obj.time_in.time() > obj.shift_start_time:
                obj.status = 'L'  
            else:
                obj.status = 'P' 
            obj.save()

        if obj.time_in and obj.time_out:
            ti = obj.time_in
            to = obj.time_out
            hours = (to - ti).total_seconds() / 3600
            obj.hours = hours
            obj.shift_end_time = user_shift.end_time
            if obj.hours < 8 or obj.time_out.time() < obj.shift_end_time and obj.status != 'L':
                obj.status = 'E' 
                obj.save()
        else:
            obj.hours = 0
            
        df_hours.append(obj.hours)

        obj.hours = convert_hours_to_hours_mins(obj.hours)

    df = read_frame(qs)
    df["hours"] = df_hours

    print(df)
    
    sns.barplot(data=df, x='date', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    
    if admin:
        plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
    else:
        plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
    plt.close()

    return qs
	

#used
def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0

		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)


	df = read_frame(qs)	
	df['hours']=df_hours
	df['employee']=df_username


	sns.barplot(data=df,x='employee',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin 


def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)

def employees_not_present_today():
    total = total_number_employees()
    present = employees_present_today()
    return total - present

def plot_employees_presence():
    present_count = employees_present_today()
    non_present_count = employees_not_present_today()

    labels = 'Present', 'Not Present'
    sizes = [present_count, non_present_count]
    colors = ['#4CAF50', '#F44336']  # Updated colors for a modern look
    explode = (0.1, 0)  # "Explode" the first slice for emphasis

    # Set up figure and adjust layout
    plt.figure(figsize=(10, 8))
    rcParams.update({'font.size': 14})  # Standardize font size

    # Create the pie chart with new attributes
    patches, texts, autotexts = plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', pctdistance=0.85, shadow=False, startangle=90)

    # Customize the design of autotexts and texts
    for text in autotexts:
        text.set_color('black')  # Enhance readability with a darker color
        text.set_fontsize(24)  # Larger font size for percentage labels
    
    # Draw a circle at the center of pie to create a donut-like appearance
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.axis('equal')  # Ensure it's drawn as a circle

    # Add a legend, title, and save the figure
    plt.legend(patches, labels, loc='best')  # Use a legend instead of labels on the slices
    plt.title('Employee Present on ' + str(datetime.date.today()), fontsize=32)  # Set title with a larger font size
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/present_today/1.png', dpi=300)
    plt.close()

#used	
def this_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_this_week, date__lte=today)
    
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        formatted_date = date.strftime('%d-%m')  # Format date to day-month
        str_dates.append(formatted_date)
        daily_qs = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(daily_qs))

    while cnt < 7:
        date = monday_of_this_week + datetime.timedelta(days=cnt)
        formatted_date = date.strftime('%d-%m')
        str_dates_all.append(formatted_date)
        if formatted_date in str_dates:
            idx = str_dates.index(formatted_date)
            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)
        cnt += 1

    df = pd.DataFrame({
        "date": str_dates_all,
        "Number of employees": emp_cnt_all
    })
    
    sns.lineplot(data=df, x='date', y='Number of employees')
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
    plt.close()


#used
def last_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
    
    str_dates = []
    emp_count = []

    str_dates_all = []
    emp_cnt_all = []
    cnt = 0
    
    for obj in qs:
        date = obj.date
        formatted_date = date.strftime('%d-%m')  # Format to show day and month
        str_dates.append(formatted_date)
        day_qs = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(day_qs))

    while cnt < 7:  # Ensure it covers the whole week
        date = monday_of_last_week + datetime.timedelta(days=cnt)
        formatted_date = date.strftime('%d-%m')
        str_dates_all.append(formatted_date)
        if formatted_date in str_dates:
            idx = str_dates.index(formatted_date)
            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)
        cnt += 1

    df = pd.DataFrame({
        "date": str_dates_all,
        "Number of employees": emp_cnt_all
    })
    
    sns.lineplot(data=df, x='date', y='Number of employees')
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
    plt.close()

def this_month_emp_count_vs_date():
    today = datetime.date.today()
    first_day_of_this_month = today.replace(day=1)
    last_day_of_this_month = today.replace(day=28) + datetime.timedelta(days=4)  # ensure it covers the end of the month
    last_day_of_this_month = last_day_of_this_month - datetime.timedelta(days=last_day_of_this_month.day)

    qs = Present.objects.filter(date__range=[first_day_of_this_month, last_day_of_this_month]).exclude(status='A')
    
    # Initialize dictionary to store attendance counts per "labelled" week
    weekly_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    # Calculate the total number of days in the month
    total_days = (last_day_of_this_month - first_day_of_this_month).days + 1

    # Loop through each day of the month, assign each day to a week from 1 to 4
    for i in range(total_days):
        day = first_day_of_this_month + datetime.timedelta(days=i)
        week_of_month = (i // 7) + 1  # Determine week number by dividing day number by 7
        if week_of_month > 4:
            week_of_month = 4  # Assign all days after the 28th to week 4

        # Filter records for each specific day
        daily_records = qs.filter(date=day)
        weekly_counts[week_of_month] += daily_records.count()

    # Prepare data for plotting
    weeks = [f"Week {week}" for week in weekly_counts.keys()]
    counts = [weekly_counts[week] for week in weekly_counts.keys()]

    plt.figure(figsize=(10, 6))  # Example dimensions in inches (width, height)

    df = pd.DataFrame({
        "Week": weeks,
        "Number of employees": counts
    })

    sns.lineplot(data=df, x='Week', y='Number of employees')
    plt.ylabel('Number of Employees')
    plt.xlabel('Week of the Month')
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_month/1.png')
    plt.close()
    
def last_month_emp_count_vs_date():
    today = datetime.date.today()
    first_day_of_this_month = today.replace(day=1)
    first_day_of_last_month = (first_day_of_this_month - datetime.timedelta(days=1)).replace(day=1)
    last_day_of_last_month = first_day_of_this_month - datetime.timedelta(days=1)

    qs = Present.objects.filter(date__range=[first_day_of_last_month, last_day_of_last_month]).exclude(status='A')
    
    # Initialize dictionary to store attendance counts per "labelled" week
    weekly_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    # Calculate the total number of days in the month
    total_days = (last_day_of_last_month - first_day_of_last_month).days + 1

    # Loop through each day of the month, assign each day to a week from 1 to 4
    for i in range(total_days):
        day = first_day_of_last_month + datetime.timedelta(days=i)
        week_of_month = (i // 7) + 1  # Determine week number by dividing day number by 7
        if week_of_month > 4:
            week_of_month = 4  # Assign all days after the 28th to week 4

        # Filter records for each specific day
        daily_records = qs.filter(date=day)
        weekly_counts[week_of_month] += daily_records.count()

    # Prepare data for plotting
    weeks = [f"Week {week}" for week in weekly_counts.keys()]
    counts = [weekly_counts[week] for week in weekly_counts.keys()]

    plt.figure(figsize=(10, 6))  # Example dimensions in inches (width, height)

    df = pd.DataFrame({
        "Week": weeks,
        "Number of employees": counts
    })

    sns.lineplot(data=df, x='Week', y='Number of employees')
    plt.ylabel('Number of Employees')
    plt.xlabel('Week of the Month')
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_month/1.png')
    plt.close()

# Create your views here.
def home(request):
	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")
		return render(request,'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')


	else:
		form=usernameForm()
		return render(request,'recognition/add_photos.html', {'form' : form})

def mark_your_attendance(request):
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	
	
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 300)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')

	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	
	while(True):
		
		frame = vs.read()
		
		frame = imutils.resize(frame ,width = 800)
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray_frame,0)
		
		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
							
			(pred,prob)=predict(face_aligned,svc)
			
			if(pred!=[-1]):
				
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

                # Define the background color and create the background image
				background_color = (0, 0, 0)
				background_image = np.full((150, 900, 3), background_color, dtype=np.uint8)

				# Add text to the image
				font = cv2.FONT_HERSHEY_SIMPLEX
				text_lines = [
					"Hi " + person_name.title() + ". " + "Your Attendance Check-In is marked.",
					"Please press q to exit."
				]  # List of text lines to draw
				text_color = (255, 255, 255)  # White color for the text

				# Initial vertical position for the first line of text
				text_y = 50  # Start drawing the first line at y = 50

				# Draw each line of text
				for line in text_lines:
					text_size = cv2.getTextSize(line, font, 1, 2)[0]
					text_x = (background_image.shape[1] - text_size[0]) // 2  # Center the text horizontally
					cv2.putText(background_image, line, (text_x, text_y), font, 1, text_color, 2, cv2.LINE_AA)
					text_y += text_size[1] + 10  # Update y position for the next line (adding a gap of 10 pixels)


                # Display the image in a window for 10 seconds
				cv2.imshow("Face Detected", background_image)
				#update_attendance_in_db_in(present)
				#cv2.waitKey(5000)

				#vs.stop()  # Stop the video stream
				#cv2.destroyAllWindows()  # Close all windows
				#return redirect('home')  # Redirect to the desired page after detecting a face

                
                #break  # Exit the for loop after detecting a face
			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	update_attendance_in_db_in(present)
	return redirect('home')


def mark_your_attendance_out(request):
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	
		
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 300)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False
	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	
	while(True):
		
		frame = vs.read()
		
		frame = imutils.resize(frame ,width = 800)
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray_frame,0)
		
		
		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
					
			
			(pred,prob)=predict(face_aligned,svc)
			
			
			if(pred!=[-1]):
				
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

				# Define the background color and create the background image
				background_color = (0, 0, 0)
				background_image = np.full((150, 900, 3), background_color, dtype=np.uint8)

				# Add text to the image
				font = cv2.FONT_HERSHEY_SIMPLEX
				text_lines = [
					"Bye " + person_name.title() + ". " + "Your Attendance Check-Out is marked.",
					"Please press q to exit."
				]  # List of text lines to draw
				text_color = (255, 255, 255)  # White color for the text

				# Initial vertical position for the first line of text
				text_y = 50  # Start drawing the first line at y = 50

				# Draw each line of text
				for line in text_lines:
					text_size = cv2.getTextSize(line, font, 1, 2)[0]
					text_x = (background_image.shape[1] - text_size[0]) // 2  # Center the text horizontally
					cv2.putText(background_image, line, (text_x, text_y), font, 1, text_color, 2, cv2.LINE_AA)
					text_y += text_size[1] + 10  # Update y position for the next line (adding a gap of 10 pixels)

                # Display the image in a window for 10 seconds
				cv2.imshow("Face Detected", background_image)
				#update_attendance_in_db_out(present)
				#cv2.waitKey(5000)

				#vs.stop()  # Stop the video stream
				#cv2.destroyAllWindows()  # Close all windows
				#return redirect('home')  # Redirect to the desired page after detecting a face
			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	update_attendance_in_db_out(present)
	return redirect('home')

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('jpg', 'jpeg', 'png'))]


@login_required
def train(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')

    start_time = time.time()  # Start the timer

    training_dir = 'face_recognition_data/training_dataset'
    count = 0
    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            count += 1

    X = []
    y = []
    i = 0

    for person_name in os.listdir(training_dir):
        print(str(person_name))
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                X.append((face_recognition.face_encodings(image)[0]).tolist())
                y.append(person_name)
                i += 1
            except:
                print("removed")
                os.remove(imagefile)

    targets = np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    np.save('face_recognition_data/classes.npy', encoder.classes_)
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X1, y)
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f)

    vizualize_Data(X1, targets)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    messages.success(request, f'Training Complete. Time taken: {elapsed_time:.2f} seconds.')

    return render(request, "recognition/train.html")


@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')


@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	plot_employees_presence()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	this_month_emp_count_vs_date()
	last_month_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})

@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None


	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)
				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')
	else:
			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None
	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						#print("inside qs is None")
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')
	else:
			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})


@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
				messages.warning(request, f'Invalid date selection.')
				return redirect('view-my-attendance-employee-login')
			else:
				time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')				
				if (len(time_qs)>0 or len(present_qs)>0):
					qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
					return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
				else:
					messages.warning(request, f'No records for selected duration.')
					return redirect('view-my-attendance-employee-login')
	else:
		form=DateForm_2()
		return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})

def employee_list(request):
    employees = User.objects.all().prefetch_related('shift_set')
    return render(request, 'recognition/employee_list.html', {'employees': employees})

def employee_edit(request, id=None):
    if id:
        employee = get_object_or_404(User, id=id)
        try:
            shift = Shift.objects.get(user=employee)
        except Shift.DoesNotExist:
            shift = None
    else:
        employee = User()
        shift = None

    if request.method == 'POST':
        user_form = EmployeeForm(request.POST, instance=employee)
        shift_form = ShiftEdit(request.POST, request.FILES, instance=shift)

        if user_form.is_valid() and shift_form.is_valid():
            # Save user
            saved_user = user_form.save(commit=False)
            # Save shift
            saved_shift = shift_form.save(commit=False)
            saved_shift.user = saved_user
            saved_shift.save()

            return redirect('employee_list')
    else:
        user_form = EmployeeForm(instance=employee)
        shift_data = {'user': employee} if shift is None else {}
        shift_form = ShiftEdit(instance=shift, initial=shift_data)

    return render(request, 'recognition/employee_form.html', {
        'user_form': user_form,
        'shift_form': shift_form
    })

def employee_delete(request, id):
	employee = get_object_or_404(User, id=id)
	if request.method == 'POST':
		employee.delete()
		return redirect('employee_list')
	return render(request, 'recognition/employee_confirm_delete.html', {'employee': employee})

@login_required
def calendar_view(request):
    return render(request, 'recognition/calendar.html')

@login_required
def shifts_api(request):
    shift_colors = {
        'Shift A (04:00 - 12:00)': '#ffadad',  # Light red
        'Shift B (12:00 - 20:00)': '#ffd6a5',  # Light orange
        'Shift C (20:00 - 04:00)': '#caffbf',  # Light green
        'Day Off': '#9bf6ff',  # Light blue
    }
    shifts = ShiftCalendar.objects.all()
    shift_data = []
    for shift in shifts:
        user_names = ', '.join(user.username for user in shift.users.all())
        shift_type = shift.get_shift_type_display()
        color = shift_colors.get(shift_type, '#bdb2ff')  # Default color if no match
        shift_entry = {
            'title': f"{shift_type} - {user_names}",
            'start': shift.date.isoformat(),
            'allDay': True,
            'id': shift.id,
            'color': color  # Add color to each shift entry
        }
        shift_data.append(shift_entry)
    return JsonResponse(shift_data, safe=False)

@login_required
def add_shift(request):
    if not request.user.is_superuser:
        return HttpResponse('', status=204)
    initial_data = {}
    # Check if a date is provided in the URL and use it to prepopulate the form
    date_str = request.GET.get('date')
    if date_str:
        try:
            # Parse the string to a date object (ensure the format matches the input)
            initial_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            initial_data['date'] = initial_date
        except ValueError:
            # Handle the error in case of an invalid date format
            pass

    if request.method == 'POST':
        form = ShiftForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('calendar')  # Ensure 'calendar' is a valid URL name in your url patterns
    else:
        form = ShiftForm(initial=initial_data)  # Pass the initial data to the form when GET request

    return render(request, 'recognition/add_shift.html', {'form': form})

@login_required
def edit_shift(request, shift_id):
    if not request.user.is_superuser:
        return HttpResponse('', status=204)
    shift = get_object_or_404(ShiftCalendar, pk=shift_id)
    if request.method == 'POST':
        if 'delete' in request.POST:  # Check if deleting
            shift.delete()
            return redirect('calendar')
        else:
            form = ShiftForm(request.POST, instance=shift)
            if form.is_valid():
                form.save()
                return redirect('calendar')
    else:
        form = ShiftForm(instance=shift)
    return render(request, 'recognition/edit_shift.html', {'form': form, 'shift_id': shift_id})