from flask import Flask, render_template, request, send_from_directory, send_file
from flask_restx import reqparse, Api, Resource
import uuid
import numpy as np
from run_animate_from_file import run_animation
import os


app = Flask(__name__)
api = Api(app, doc='/docs', prefix='/api')

# WEBSITE ########################################################################################
@app.route('/')
def index():
    return render_template("index.html")

# SERVER API #####################################################################################
class Job():
    def __init__(self, points):
        self.id = str(uuid.uuid4())
        self.points = points
        self.is_done = False
    
    def get_id(self):
        return self.id
    
    def get_points(self):
        return self.points

    def get_job_info(self):
        return {
            "job_id": self.id,
            "is_done": self.does_video_exist(),
        }
    
    def set_job_status_done(self):
        self.is_done = True
    
    def does_video_exist(self):
        filepath = "media/videos/animate_from_file/1440p60/VectorRender.mp4"
        return os.path.exists(filepath)

job_dict = {}

job_creation_args = reqparse.RequestParser()
job_creation_args.add_argument('points', type=list, location='json', required=True);

@api.route('/job')
class JobApi(Resource):
    def get(self):
        job_id = request.args.get('job_id')
        job = job_dict[job_id]
        return job.get_job_info(), 200

    @api.expect(job_creation_args)
    def post(self):
        args = job_creation_args.parse_args()
        points = args['points']

        new_job = Job(points);
        new_job_id = new_job.get_id();
        job_dict[new_job_id] = new_job

        points_ndarray = np.array(
            [[point["x"], point["y"]] for point in points]
        )

        np.save("points/" + new_job_id + ".npy", points_ndarray);

        response = {
            "job_id": new_job_id
        }

        run_animation(new_job_id);
        return response, 201

@api.route('/job/video')
class videoApi(Resource):
    def get(self):
        job_id = request.args.get('job_id')
        video_path = "media/videos/animate_from_file/1440p60/VectorRender.mp4"
        return send_file(video_path, mimetype='video/mp4')

if __name__ == "__main__":
    app.run(debug=True)