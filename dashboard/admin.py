from django.contrib import admin
from .models import *

admin.site.register(Patients)
admin.site.register(Referrals)
admin.site.register(ODReferrals)
admin.site.register(Encounters)
