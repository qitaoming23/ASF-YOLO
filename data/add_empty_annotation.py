import os
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def create_empty_annotation(filename, width, height, depth=3):
    root = Element('annotation')
    
    # Filename
    filename_elem = SubElement(root, 'filename')
    filename_elem.text = filename
    
    # Source
    source = SubElement(root, 'source')
    annotation = SubElement(source, 'annotation')
    annotation.text = 'ESRI ArcGIS Pro'
    
    # Size
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = str(depth)
    
    # Generate pretty XML
    xml_str = tostring(root, 'utf-8')
    pretty_xml = parseString(xml_str).toprettyxml(indent="    ")
    return pretty_xml

def check_and_generate_annotations(images_dir, annotations_dir):
    images = [img for img in os.listdir(images_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    for img in images:
        img_path = os.path.join(images_dir, img)
        annotation_path = os.path.join(annotations_dir, os.path.splitext(img)[0] + '.xml')
        
        if not os.path.exists(annotation_path):
            with Image.open(img_path) as im:
                width, height = im.size
                depth = len(im.getbands())  # Assuming the image is either grayscale (1) or RGB (3)
                
            xml_content = create_empty_annotation(img, width, height, depth)
            with open(annotation_path, 'w') as f:
                f.write(xml_content)
            print(f'Generated empty annotation for {img}')

# Change these paths to your actual directories
images_dir = '/home/wdblink/Dataset/RGB-DSM/images'
annotations_dir = '/home/wdblink/Dataset/RGB-DSM/labels'

check_and_generate_annotations(images_dir, annotations_dir)

