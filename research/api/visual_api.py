""" DCGAN implments for noise generate cartoon image"""


import os
import torch
from django.shortcuts import render
from rest_framework.views import APIView

# dcgan
from utils.test import generate


# model
def index(request):
  """Get the image based on the base64 encoding or url address
          and do the pencil style conversion
  Args:
      request: Post request in url.
      - image_code: 64-bit encoding of images.
      - url:        The URL of the image.
  Returns:
      Base64 bit encoding of the image.
  Notes:
      Later versions will not return an image's address,
      but instead a base64-bit encoded address
  """
  return render(request, "index.html")


class CartoonSister(APIView):
  """ use dcgan generate animel sister
  """

  @staticmethod
  def get(request):
    """ Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Returns:
        Base64 bit encoding of the image.
    Notes:
        Later versions will not return an image's address,
        but instead a base64-bit encoded address
    """
    ret = {
      "status_code": 20000,
      "message": None,
      "image": None}
    return render(request, "dcgan.html", ret)

  @staticmethod
  def post(request):
    """ Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Returns:
        Base64 bit encoding of the image.
    Notes:
        Later versions will not return an image's address,
        but instead a base64-bit encoded address
    """
    # Get the url for the image
    cartoon_path = "./static/cartoon_sister.png"
    generate()

    ret = {
      "status_code": 20000,
      "message": "OK",
      "image": cartoon_path}
    return render(request, "dcgan.html", ret)
