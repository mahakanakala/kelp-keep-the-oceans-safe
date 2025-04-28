# Use the official Python image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for uploads
RUN mkdir -p uploads/oil_spills uploads/garbage

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create a script to run migrations and start the app
RUN echo '#!/bin/bash\n\
python database.py\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0' > /app/start.sh

RUN chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"] 