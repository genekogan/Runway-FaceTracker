import runway
from runway.data_types import number, text, image, array, image_bounding_box
from example_model import FaceTracker

setup_options = {
}
@runway.setup(options=setup_options)
def setup(opts):
    model = FaceTracker(opts)
    return model

@runway.command(name='find_faces',
                inputs={ 'input': image(description="The input image to analyze") },
                outputs={ 'ids': array(number, description="IDs of found faces"), 'boxes': array(image_bounding_box, description="bounding boxes of found faces") },
                description='Look for faces in the image')
def find_faces(model, args):

    output = model.process(args['input'])
    
    return {
        'ids': [o["index"] for o in output],
        'boxes': [o["box"] for o in output]
    }

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000, debug=True)
