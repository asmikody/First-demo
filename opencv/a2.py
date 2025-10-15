import cv2
file=input("Enter the filename: ")
file_name=rf"{file}"
img=cv2.imread(file_name)
image=cv2.resize(img,(400,400))
op=int(input("enter 1 for draw a line\nEnter 2 for draw a circle\nenter 3 for draw a retangle\nEnter 4 for put text on it\n"))
if op==1:
    x1=int(input("enter the value of x1: "))
    x2=int(input("enter the value of x2: "))
    x3=int(input("enter the value of x3: "))
    x4=int(input("enter the value of x4: "))
    cv2.line(image,(x1,x2),(x3,x4),(255,0,0),1)
    
    num=int(input("enter 1 for show image\nenter 2 for save the image"))
    if num==1:
        cv2.imshow("line on image",image)
    elif num==2:
        name=input("enter the file name(.jpg or .png foramt): ")
        cv2.imwrite(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif op==2:
    x1=int(input("enter the value of x1 for center: "))
    x2=int(input("enter the value of x2 for center: "))
    r=int(input("enter the value of radious"))
    cv2.circle(image,(x1,x2),r,(255,0,0),1)
    num=int(input("enter 1 for show image\nenter 2 for save the image"))
    if num==1:
        cv2.imshow("line on image",image)
    elif num==2:
        name=input("enter the file name(.jpg or .png foramt): ")
        cv2.imwrite(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif op==3:
    x1=int(input("enter the value of x1: "))
    x2=int(input("enter the value of x2: "))
    x3=int(input("enter the value of x3: "))
    x4=int(input("enter the value of x4: "))
    cv2.rectangle(image,(x1,x2),(x3,x4),(0,0,0),3)
    num=int(input("enter 1 for show image\nenter 2 for save the image"))
    if num==1:
        cv2.imshow("line on image",image)
    elif num==2:
        name=input("enter the file name(.jpg or .png foramt): ")
        cv2.imwrite(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif op==4:
    x1=int(input("enter the value of x1: "))
    x2=int(input("enter the value of x2: "))
    txt=input("enter the text ")
    cv2.putText(image,txt,(x1,x2),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,0,0),3)
    num=int(input("enter 1 for show image\nenter 2 for save the image"))
    if num==1:
        cv2.imshow("line on image",image)
    elif num==2:
        name=input("enter the file name(.jpg or .png foramt): ")
        cv2.imwrite(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    exit(0)