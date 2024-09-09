=== Installation Instructions ===

Step 1: Install the Client-side service

Navigate to the folder containing phishlang.deb and enter "sudo dpkg -i phishlang.deb"

OR

Double click on the .deb file and install it. 

Note: It may take some time (2-5 mins on a 100Mbps connection) as it is downloading the dependencies. At this point the installer will show the status as “Preparing” and might seem stuck, please don't abort the installation - Tested on Ubuntu 22.04 LTS/24.04 LTS

Post installation Phishlang will run in your background. You can check if it is running by typing:

"sudo systemctl status phishlang.service" in your terminal.


Step 2: Install the Web extension

On any Chromium based browser (Google Chrome/Brave/Microsoft Edge etc.) goto Extensions -> Turn on Developer mode. Then drag and drop the phishlang_extension.crx file into the Extension window.
