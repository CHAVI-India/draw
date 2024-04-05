from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth import authenticate, login, logout
from .forms import LoginForm, SignupForm
from django.contrib.auth.forms import UserCreationForm
from django.middleware.csrf import get_token    
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from datetime import datetime
from .models import *
import subprocess
import shutil
import zipfile
import os
from django.conf import settings
from django.http import HttpResponseRedirect
from slices import *
import json
import threading
from django.http import FileResponse


parts=['brain','lung','prime']
model_to_part={
    'nnUNet_for_brain':'brain',
    'nnUNet_for_lung':'lung',
    'nnUNet_for_prime':'prime',
               }
part_to_model={
    'brain':'nnUNet_for_brain',
    'lung':'nnUNet_for_lung',
    'prime':'nnUNet_for_prime',
}

db_json_path='./data/nnUNet_raw/Dataset720_TSPrime/db.json'


# login view


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            print("he is there")
            return redirect('home')  # Redirect to the home page after successful login
        else:
            print("invalid login")
            # Handle invalid login
            return render(request, 'login.html')
    else:
        print("didn't work")
        return render(request, 'login.html')
    
    
# signup view

def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            print("registered")
            username = form.cleaned_data['username']
            uname=form.cleaned_data['name']
            uemail=form.cleaned_data['email']   
            user = User.objects.filter(username=username).first()
            if user:
                newuser=autoseg_user.objects.create(name=uname,username=username,email=uemail)
                user_folder = os.path.join(settings.MEDIA_ROOT, "users_data")
                user_folder=os.path.join(user_folder,username)
                os.makedirs(user_folder, exist_ok=True)
                for x in parts:
                    parts_folder=os.path.join(user_folder,x)
                    os.makedirs(parts_folder, exist_ok=True)
            return redirect('login_view')  # Redirect to the login page after successful signup
        else:
            print("reg error")
            # Handle invalid form submission
            return render(request, 'login.html') 
    else:
        return render(request, 'login.html')


# logout view

def logout_view(request):
    logout(request)
    return redirect('login_view')  # Redirect to the login page after successful logout


# home page view

@login_required(login_url='login_view')
def home(request):
    template = loader.get_template('home.html')
    return render(request, 'home.html')





def make_prediction(input_folder,output_folder,body_part):
    # /media/neeraj/E/Documents/BTP/autoseg_web is pwd
    f_input_folder='./data/nnUNet_raw/Dataset720_TSPrime/imagesTr'
    virtual_env_path = '../watchtower/bin/activate'
    if body_part=="brain":
        command = f'export RESULTS_FOLDER="/media/neeraj/E2/Documents/BTP/nnUNet-1.7.1/brats/results" && nnUNet_predict -i {input_folder} -o {output_folder} -t 001 -m 3d_fullres'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    elif body_part=="lung":
        command = f'export RESULTS_FOLDER="/media/neeraj/E2/Documents/BTP/nnUNet-1.7.1/results" && nnUNet_predict -i {input_folder} -o {output_folder} -t 106 -m 2d'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
       
    elif body_part=="prime":
        command = (
    f'export nnUNet_raw="./data/nnUNet_raw"; '
    f'export nnUNet_preprocessed="./data/nnUNet_preprocessed"; '
    f'export nnUNet_results="./data/nnUNet_results" && '
    f'nnUNetv2_predict -i {f_input_folder} -o {output_folder} -f 0 -d 720 -c 3d_fullres'
)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return


def compute_slices(input_file_path,output_folder,output_file_path,bpart,ffol_name):
                       sfol_name=os.path.join(output_folder,ffol_name)
                       os.makedirs(sfol_name, exist_ok=True)
                       save_axial_slice_images_with_overlay(input_file_path,output_file_path,os.path.join(output_folder, ffol_name))
                       print("overlay images done*-----")



     
def check_status(job_data,username):
    user_folder = os.path.join(settings.MEDIA_ROOT, "users_data", username)
    for x in job_data:
        if x.status=="completed":
            continue
        bp=model_to_part[x.model_used]
        body_part_folder = os.path.join(user_folder, bp)
        timestamp_folder = os.path.join(body_part_folder, x.time_stamp)
        output_folder = os.path.join(timestamp_folder, "output")
        t_input_folder = os.path.join(timestamp_folder, "input")
        tl=os.listdir(t_input_folder)
        inc=len(tl)
        outc=0
        for file_name in os.listdir(output_folder):
                    if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                         outc+=1
        if inc==outc:
            with open(db_json_path, 'r') as file:
                json_data = json.load(file)
            directory_to_sample = {}
            for entry in json_data:
                directory = entry['DICOMRootDir'].split('/')[-1]  # Get the directory name from the path
                sample_number = entry['SampleNumber']
                directory_to_sample[directory] = sample_number
            for dicom_dir_name in tl:
                dicom_dir=os.path.join(t_input_folder,dicom_dir_name)
                templ=os.listdir(dicom_dir)
                last_file = templ[-1]
                nparts = last_file.split(".")
                nparts[-2] = "1"
                nparts[-1] = "0"
                nparts.pop(-3)
                fol_name = nparts[0]
                rt_file_name = ".".join(nparts)
                print("**************")
                print(dicom_dir)
                print(rt_file_name)
                input_folder='./data/nnUNet_raw/Dataset720_TSPrime/imagesTr'
                nifti_file_path=""
                nfilename="seg_"+str(directory_to_sample[fol_name])+".nii.gz"
                nifti_file_path=os.path.join(output_folder,nfilename)
                infilename="seg_"+str(directory_to_sample[fol_name])+"_0000.nii.gz"
                infilepath=os.path.join(input_folder,infilename)
                compute_slices(infilepath,output_folder,nifti_file_path,"brain",fol_name)
                print("in here")
                print(nifti_file_path)
                print(dicom_dir)
                print(output_folder)
                print("-------------------")
                convert_nifti_to_rtstruct(nifti_file_path, dicom_dir, output_folder,rt_file_name)
                #saving the rt_struct file in input dicom folder and zipping it for user to download
                rt_file_path=os.path.join(output_folder,rt_file_name)
                rt_file_path+=".dcm"
                print("rtfilepath is ",rt_file_path)
                shutil.copy(rt_file_path, dicom_dir)
                out_zip=dicom_dir
                out_zip+=".zip"
                with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(dicom_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            print("the file path ",file_path)
                            zipf.write(file_path, os.path.relpath(file_path, dicom_dir))
            job = Job.objects.get(pk=x.id)
            # Update the status attribute
            job.status = "completed"
            f_input_folder='./data/nnUNet_raw/Dataset720_TSPrime'
            command = f'rm -r {f_input_folder}'
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Save the object to update it in the database
            job.save()


# upload images view

@login_required(login_url='login_view')
def upload_images(request):
    if request.method == 'POST':
        selected_body_part = request.POST.get('selected-model')
        study_description = request.POST.get('description')
        current_user = request.user
        username = current_user.username

        # Handle uploaded zip file
        uploaded_zips = request.FILES.getlist('zip_files')
        if len(uploaded_zips) != 0:
            # Create timestamp for folder name
            print("zip found")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Create folder paths
            user_folder = os.path.join(settings.MEDIA_ROOT, "users_data", username, selected_body_part, timestamp)
            upload_folder = os.path.join(user_folder, "input")
            output_folder = os.path.join(user_folder, "output")
            
            # Create directories if they don't exist
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)
            
            print(upload_folder)
            # Extract zip file
            for uploaded_zip in uploaded_zips:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(upload_folder)
            command = (
    f'export nnUNet_raw="./data/nnUNet_raw"; '
    f'export nnUNet_preprocessed="./data/nnUNet_preprocessed"; '
    f'export nnUNet_results="./data/nnUNet_results" && '
    f'python main.py preprocess -d {upload_folder} -i 720 -n TSPrime --start 1 --only-original'
)
            
        
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            make_prediction(upload_folder,output_folder,selected_body_part)
            # Create a new job entry in the database
            job = Job.objects.create(
                time_stamp=timestamp,
                description=study_description,
                status="Pending",  # Set the initial status
                model_used=part_to_model[selected_body_part],
                user=autoseg_user.objects.get(username=username)
            )

            print("Job entry created successfully")

            return redirect('home')  # Redirect to a success page after uploading
    return redirect('home')

@login_required(login_url='login_view')
def results(request):
    # Fetch job data for the current user
    current_user = request.user
    user=autoseg_user.objects.get(username=current_user.username)
    job_data = Job.objects.filter(user=user)
    
    check_status(job_data,current_user.username)
    
    # def run_check_status():
    #     check_status(job_data, current_user.username)
    # # Create and start a new thread
    # thread = threading.Thread(target=run_check_status)
    # thread.start()
    
    j_data=[]
    for x in job_data:
        j_data.append({
                            'description': x.description,
                            'timestamp': x.time_stamp,
                            'status': x.status,
                            'j_id':x.id,
                        })
    # Render the HTML template with the filtered job data
    j_data_a=[]
    idx=1
    for x in j_data:
        j_data_a.append((idx,x))
        idx+=1    
    return render(request, 'results.html', {'job_data': j_data_a})


@login_required(login_url='login_view')
def download_file(request, job_id,file_name):
    job_det=Job.objects.get(id=job_id)
    timestamp=job_det.time_stamp
    body_part=model_to_part[job_det.model_used]
    print(timestamp)
    # Construct the file path
    username=request.user.username
    file_path = os.path.join(settings.MEDIA_ROOT, "users_data", username, body_part, timestamp, "output", file_name)

    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type='application/octet-stream')
        
    # Set the Content-Disposition header to force download
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    
    return response

@login_required(login_url='login_view')
def download_folder(request, job_id,file_name):
    job_det=Job.objects.get(id=job_id)
    timestamp=job_det.time_stamp
    body_part=model_to_part[job_det.model_used]
    print(timestamp)
    # Construct the file path
    username=request.user.username
    nparts = file_name.split(".")
    zip_file_name=nparts[0]+".zip"
    print("zip file path to download is ",zip_file_name)
    file_path = os.path.join(settings.MEDIA_ROOT, "users_data", username, body_part, timestamp, "input", zip_file_name)
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type='application/octet-stream')
        
    # Set the Content-Disposition header to force download
    response['Content-Disposition'] = f'attachment; filename="{zip_file_name}"'
    
    return response

@login_required(login_url='login_view')
def image_slider_view(request,job_id,file_name):
    job_det=Job.objects.get(id=job_id)
    timestamp=job_det.time_stamp
    body_part=model_to_part[job_det.model_used]
    parts = file_name.split(".")
    fol_name = parts[0]
    # Construct the file path
    username=request.user.username

    # Assuming the images are stored in a directory named 'slices' inside MEDIA_ROOT
    images_dir = os.path.join(settings.MEDIA_ROOT,'users_data',username,body_part,timestamp,"output",fol_name)

    
    # Get a list of image filenames in the directory
    image_filenames = os.listdir(images_dir)
    image_filenames = sorted(image_filenames, key=lambda x: int(x.split('.')[0]))
    
    
    # Generate URLs for each image
    image_urls = [os.path.join(settings.MEDIA_URL,'users_data',username,body_part,timestamp,"output",fol_name, filename) for filename in image_filenames]
    print("going to render image") 
    return render(request, 'rtstruct.html', {'image_urls': image_urls})

@login_required(login_url='login_view')
def job_result(request, job_id):
    job_det=Job.objects.get(id=job_id)
    j_data= {
        "time_stamp":job_det.time_stamp,
        "status":job_det.status,
        "description":job_det.description,
        "model_used":job_det.model_used,
        "job_id":job_det.id,
    }
    user_folder = os.path.join(settings.MEDIA_ROOT, "users_data", request.user.username)
    # Initialize an empty list to store file information
    file_data = []
    body_part_folder = os.path.join(user_folder, model_to_part[job_det.model_used])
    timestamp_folder = os.path.join(body_part_folder, job_det.time_stamp)
    output_folder = os.path.join(timestamp_folder, "output")
    if os.path.exists(output_folder):
                for file_name in os.listdir(output_folder):
                    # Append file information to the list
                    if file_name.endswith('.dcm') :
                        # Append file information to the list
                        file_data.append(file_name)

    return render(request, 'view_details.html', {'job': j_data,'file_data':file_data})