import uuid
import datetime
from lxml import etree as ET
import csv
from copy import copy


def csv2xml(model_name, v, input_xml_path, csv_input_paths, orientations):
    from tqdm import tqdm

    def inBounds(point, tl, br):
        x, y = point
        left, top = tl
        right, bottom = br
        if left < x < right and top < y < bottom:
            return True
        return False

    # check if these two boxes have any overlap
    def boxOverlap(bbox1, bbox2):
        # check box2 in box1
        tl = (bbox1['x0'], bbox1['y0'])
        br = (bbox1['x1'], bbox1['y1'])
        for point in [(bbox2['x0'], bbox2['y0']), (bbox2['x1'], bbox2['y1'])]:
            if inBounds(point, tl, br):
                return True

        # check box1 in box2
        tl = (bbox2['x0'], bbox2['y0'])
        br = (bbox2['x1'], bbox2['y1'])
        for point in [(bbox1['x0'], bbox1['y0']), (bbox1['x1'], bbox1['y1'])]:
            if inBounds(point, tl, br):
                return True
        # no overlap
        return False

    def getPartialInstances(framedict):
        partials = [False for _ in framedict]
        for act_pos in range(len(partials) - 1):
            if not act_pos:
                for next_pos in range(1, len(partials[act_pos:])):
                    np = act_pos + next_pos
                    if boxOverlap(framedict[act_pos], framedict[np]):
                        partials[act_pos] = True
                        partials[np] = True
        return partials

    def indent_xml_elements(elem, head_sep, level=0):
        # i = "\n" + level*"  "
        i = "\n" + level * f"{head_sep}"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + f"{head_sep}"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent_xml_elements(elem, head_sep, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def create_framework(framedict):
        # Create framework element
        framework_element = ET.Element("framework", attrib={
            "name": Framework_data.name,
            "version": Framework_data.version,
            "species": Framework_data.species,
            "labeltype": Framework_data.labeltype,
            "labeltime": f"{datetime.datetime.now():%Y%m%d%H%M%S%f}"[:-3],
            # Get actual timestamp (only 3 ms digits instead of 6)
            "confidence": Framework_data.confidence,
            "annotator": Framework_data.annotator,
            "groundtruth": Framework_data.groundtruth
        })

        # Check
        partial_instances = getPartialInstances(framedict)  # Returns boolean list with overlapping objects == true
        # Create object elements
        for fd_id, object_data in enumerate(framedict):
            obj = ET.SubElement(framework_element, "object", attrib={
                "id": str(uuid.uuid1()),
                "real": Object_data.real,
                "species": object_data['species'],
                "length": Object_data.length,
                "labeltime": f"{datetime.datetime.now():%Y%m%d%H%M%S%f}"[:-3],
                # Get actual timestamp (only 3 ms digits instead of 6)
                "labelmode": Object_data.labelmode,
                "partial": str(partial_instances[fd_id]).lower(),
                "confidence": f"{float(object_data['confidence']):.4f}",
                "annotator": Object_data.annotator,
                "groundtruth": Object_data.groundtruth
            })
            # Create and add bounding box element to object
            if object_data['orientation'] == "Left":
                bbox_side = "lbb"
            else:
                bbox_side = "rbb"
            ET.SubElement(obj, bbox_side,
                          attrib={"x0": object_data['x0'], "y0": object_data['y0'], "x1": object_data['x1'],
                                  "y1": object_data['y1']})

        if framework_element is None:
            print("returning none")

        return framework_element

    # Initialize static framework information
    class Framework_data:
        #name = "Fish_species_detector_v1.0_Vaneeda_dockerimage"
        name = model_name
        version = str(v)
        species = "None"  # Default value if no label is given (labeltype != 'image' and  labeltype != 'all')
        labeltype = "object"  # Fish instances are detected
        confidence = "0"  # Default value if no confidence is given (labeltype != 'image' and  labeltype != 'all')
        #annotator = "fish_species_detector_v1.0_Vaneeda"
        annotator = model_name + "_v" + str(v)
        groundtruth = "false"  # Default value for groundtruth, has to be manually changed after qualitative inspection

    # Initialize static object information
    class Object_data:
        real = "true"  # All fish instances are real fish (not a man-made object resembling one)
        length = "0"  # Default length value
        labelmode = "bb"
        partial = "false"  # Default "partial" flag
        annotator = Framework_data.annotator  # Same value as framework
        groundtruth = Framework_data.groundtruth  # Same value as framework

    ######################################
    ### Main script
    ######################################

    # Get input xml file and import element tree
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    if not len(root[1]):  # If no existing frame entries in xml file, exit the method
        print(f"No frames in {input_xml_path.split('/')[-1]} file, no changes applied to it.")
        return

    # Get csv file and import data to dict
    csvframedict = {}
    #print(csv_input_paths)
    for csv_pos, csvfile in enumerate(csv_input_paths):
        with open(csvfile, newline="") as csvf:
            reader = csv.DictReader(csvf)

            for row in tqdm(reader, desc=f"Importing data from {csvfile}"):
                if row['datetime'] in csvframedict:
                    csvframedict[row['datetime']].append({"x0": str(round(float(row['x0']))),
                                                          "y0": str(round(float(row['y0']))),
                                                          "x1": str(round(float(row['x1']))),
                                                          "y1": str(round(float(row['y1']))),
                                                          "species": row['label'],
                                                          "confidence": row['score'],
                                                          "orientation": orientations[csv_pos]})
                else:
                    csvframedict[row['datetime']] = [{"x0": str(round(float(row['x0']))),
                                                      "y0": str(round(float(row['y0']))),
                                                      "x1": str(round(float(row['x1']))),
                                                      "y1": str(round(float(row['y1']))),
                                                      "species": row['label'],
                                                      "confidence": row['score'], "orientation": orientations[csv_pos]}]

    # Specify the name of the outputfile and create an empty file with it
    output_xml_path = input_xml_path.split(".")[0] + "_updated.xml"
    #output_xml_path = input_xml_path
    # Set separator and encoder used to encode the strings that will be passed to the xml output file
    head_separator = "  "  # "\u0009"
    encoder = "utf-8"

    # Open and initialize the output xml file
    out_xml_fd = open(output_xml_path, "wb")
    out_xml_fd.write(r'<?xml version="1.0" encoding="utf-8"?>'.encode(encoder))
    out_xml_fd.write('\n'.encode(encoder))
    out_xml_fd.write(
        r'<deepvision xmlns:ns="http://www.deepvision.no/formats/deepvision/v1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="deepvision.xsd">'.encode(
            encoder))
    out_xml_fd.write(f'\n{head_separator}'.encode(encoder))

    # Get Info element and write it to output file
    child_cpy = copy(root[0])
    indent_xml_elements(child_cpy, head_separator, level=1)
    out_xml_fd.write(ET.tostring(child_cpy, encoding=encoder))

    # Get and store the first part of the "frames" element
    frame_string = '<frames'
    frame_elem = root.find("frames")
    for att in frame_elem.attrib:
        frame_string += f' {att}="{frame_elem.attrib[att]}"'
    frame_string += f'>\n{head_separator}{head_separator}'
    out_xml_fd.write(frame_string.encode(encoder))

    # For each child in csv except last one, update framework"
    root_len = len(root[1])
    pbar = tqdm(root_len, desc="Parsing csv data to xml file")
    if root_len > 1:
        for child in root[1][:-1]:
            child_cpy = copy(child)
            if child_cpy.attrib["time"] in csvframedict:
                framework_elem = create_framework(csvframedict[child_cpy.attrib["time"]])
                child_cpy.insert(-1, framework_elem)
                indent_xml_elements(child_cpy, head_separator, level=2)
            out_xml_fd.write(ET.tostring(child_cpy, encoding=encoder))
            pbar.update(1)
    # The tail of the last child will be set differently
    child_cpy = copy(root[1][-1])
    if child_cpy.attrib["time"] in csvframedict:
        framework_elem = create_framework(csvframedict[child_cpy.attrib["time"]])
        child_cpy.insert(-1, framework_elem)
        indent_xml_elements(child_cpy, head_separator, level=2)
        child_cpy.tail = f"\n{head_separator}"
    out_xml_fd.write(ET.tostring(child_cpy, encoding=encoder))
    pbar.update(1)
    pbar.close()

    # Add ending tags for Deepvision and Frames elements
    out_xml_fd.write("</frames>\n".encode(encoder))
    out_xml_fd.write("</deepvision>".encode(encoder))

    # Close output file
    out_xml_fd.close()

    # tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    print(f'CSV data has been exported to {output_xml_path}.')


