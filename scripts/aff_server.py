import hydra
from flask import Flask, request, jsonify
from queue import Queue

from scripts import viz_affordances

class AffordanceServer:

    def __init__(self, cfg):

        self.app = Flask(__name__)
        self.setup_route()
        self.cfg = cfg
        self.model = viz_affordances.VizAffordances(self.cfg)
        return
    

    def setup_route(self):
        '''
        Function accessed via /display_animation url.
        It receives a JSON with the path to the animation to be displayed,
        and handles the signal to the animation thread.
        '''
        @self.app.route('/affordance', methods=['POST'])
        def affordance():
            data = request.get_json()
            rgb_img = data.get("frame")
            d_img = data.get("d_img") # depth image

            if rgb_img is not None and d_img is not None:
                filename = "request"
                result = self.model.compute_aff_target(rgb_img, d_img, filename, return_data=True)
                if result is not None:
                    target_pos, no_target, world_pts = result
                    result_dict = {
                        "status": "success",
                        "message": "Affordance calculated.",
                        "target_pos": target_pos.tolist() if no_target is not False else None,
                        "no_target": bool(no_target),
                        "world_pts": [pt.tolist() for pt in world_pts]
                    }
                    return jsonify(result_dict)
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to calculate affordances.'})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid request. Missing "frame" or "d_img" field in JSON.'})


    def run_server(self):
        self.app.run(port=5000, debug=False, use_reloader=False)

        #Run this at the deployment server, in order to open up the server to the network:
        #app.run(host='0.0.0.0', port=5000, debug=False)
        
        return


@hydra.main(config_path="../config", config_name="viz_affordances")
def main(cfg):
    affordance_server = AffordanceServer(cfg)
    affordance_server.run_server()

if __name__ == "__main__":
    main()
