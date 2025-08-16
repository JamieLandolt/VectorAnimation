from flask import Flask, render_template, request
from flask_restx import reqparse, Api, Resource
import uuid
import numpy as np
from run_animate_from_file import run_animation


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
            "is_done": self.is_done
        }

job_dict = {}

job_creation_args = reqparse.RequestParser()
job_creation_args.add_argument('points', type=list, location='json', required=True);

@api.route('/job')
class JobApi(Resource):
    def get(self):
        job_id = request.args.get('job_id')

        job = job_dict[job_id] # TODO: John use a job_dict.get(job_id, default_value) here so it stops erroring
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


if __name__ == "__main__":
    app.run(debug=True)