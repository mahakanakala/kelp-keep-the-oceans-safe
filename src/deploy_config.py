import os

   # Database configuration
DB_CONFIG = {
       'development': {
           'HOST': 'localhost',
           'PORT': 5432,
           'NAME': 'kelp_oceans_db',
           'USER': 'maha_kanakala',
           'PASSWORD': '********'
       },
       'production': {
           'HOST': os.getenv('DB_HOST'),
           'PORT': os.getenv('DB_PORT'),
           'NAME': os.getenv('DB_NAME'),
           'USER': os.getenv('DB_USER'),
           'PASSWORD': os.getenv('DB_PASSWORD')
       }
   }