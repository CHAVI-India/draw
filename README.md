#  Deep learning based Radiotherapy Auto-segmentation Workflow (DRAW)


DRAW is a website which enables users to login and upload DICOM images
for segmentation and get the segmentation results and also view them.



Setting up database

login to mysql as root by "mysql -u root -p"

CREATE DATABASE autoseg;

CREATE USER 'autoseg_admin'@'localhost' IDENTIFIED BY 'Autoseg_admin@123';

GRANT ALL PRIVILEGES ON autoseg.* TO 'autoseg_admin'@'localhost';

FLUSH PRIVILEGES;


create users_data directory in media folder
