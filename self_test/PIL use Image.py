from PIL import Image
im = Image.open("C:\\Users\\admin\\Pictures\\尊重智慧財產權宣導單.jpg")
print(im.format,im.size,im.mode)
im.show()
im.save("尊重智慧財產權宣導單.pdf","pdf")