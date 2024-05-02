#  Deep learning based Radiotherapy Auto-segmentation Workflow (DRAW)


DRAW is a website which enables users to login and upload DICOM images
for segmentation and get the segmentation results and also view them.

![signup_page](https://github.com/CHAVI-India/draw/assets/101348975/e0c00af8-8998-4062-a2d8-a22f3f044a5e)
![home_page](https://github.com/CHAVI-India/draw/assets/101348975/bdcc6684-ce5c-49a3-885a-1b1aacd36e06)
![view_rtstruct](https://github.com/CHAVI-India/draw/assets/101348975/b017ddd1-71ba-46c8-aa2e-c9d67f059f5f)



Setting up database

login to mysql as root by "mysql -u root -p"

CREATE DATABASE autoseg;

CREATE USER 'autoseg_admin'@'localhost' IDENTIFIED BY 'Autoseg_admin@123';

GRANT ALL PRIVILEGES ON autoseg.* TO 'autoseg_admin'@'localhost';

FLUSH PRIVILEGES;


create users_data directory in media folder
