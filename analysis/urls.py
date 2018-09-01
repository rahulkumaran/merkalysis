from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.home, name="index"),
    # url(r'^api/(?P<pk1>\d+)/(?P<state>.+?)/$', views.get_resp, name="get_response"),
    url(r'^temp/$',views.formV,name="form")
]