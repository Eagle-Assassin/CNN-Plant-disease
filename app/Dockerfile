#This sets up the container with pyton 3.10 installed
FROM python:3.10-slim

#This copies everything in your current directory to the /app directory in the container
COPY . /app

# This sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT, or COPY instruction that follows
WORKDIR /app

#This runs pip install for all the packages listed in the requirments.txt

RUN pip install -r requirements.txt

#This tells docker to listen on port 80 at runtime. port 80 is the standard port for HTTP
EXPOSE 80

#This command creates a .straeamlit directory in the home directory of the container
RUN mkdir ~/.streamlit

#This copies your streamlit configuration file into the .Streamlit diretory you just created
RUN cp config.toml ~/.streamlit/config.toml


#This sets the default command for the container to run the app with Streamlit
ENTRYPOINT [ "streamlit","run" ]

#This command tells Strealit to run your app.py script when the container starts
CMD ["main.py"]
