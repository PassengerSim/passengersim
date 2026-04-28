
# Linux Server at Georgia Tech

If you are a student or otherwise affiliated with Georgia Tech, you can use PassengerSim on our Linux server.

## The Easy Way (or, if your local computer is Mac or Linux)

Get Ready:

- Have UV installed on your local machine, see [UV Docs](https://docs.astral.sh/uv/getting-started/installation/)
- One time only, in a command prompt / terminal run: `uv tool update-shell`. If this command doesn't work
  due to a permission error, you can try `sudo uv tool update-shell`. If that still doesn't work,
  don't worry, .
- One time only (or if told you need to update your tool), in a command prompt / terminal run:
  `uv tool install "gt-paxlab>=0.3" --index-url https://www.passengersim.com/gt-tools`

Use it:

- Be on the VPN
- In a command prompt / terminal run: `gt-paxlab`.  If this doesn't work (likely because `update-shell`
  failed), try `~/.local/bin/gt-paxlab` instead.
- It will ask you for your GT ID, password, the name of the server,
  and what directory you want to work in.
- A web browser will open with a Jupyter Lab interface running on the server.
- In the future, it will remember the ID, server name, and directory, you'll just need your password.


### Troubleshooting

- What if a browser doesn't open?  You can open a browser and navigate to the URL shown at the bottom of
  your terminal window.
- What if the connection fails?  Your Jupyter server should stay up. The next time you reconnect you'll
  be able to reconnect to the same server and pick up where you left off.


## The Hard Way (or, if your local computer is Windows)

- Be on VPN.
- Open a command prompt and run: `ssh YOUR_GT_USER_ID@SERVER_ADDRESS.ce.gatech.edu`, replacing
  `YOUR_GT_USER_ID` with your actual GT user ID and `SERVER_ADDRESS` with the server address.
  It may ask you for your password, and then you should be connected to the server.
- If necessary, update your server version of PassengerSim on the server by running:
  `uv run python /opt/pax-installers/update-paxlab.py`
- Start a JupyterLab server with a file directory locked in the directory you want to work in,
  for example if you want to work in `/mnt/cee-lgarrow/my-workspace` then run:
  `nohup uv run -p .paxlab jupyter lab --no-browser --notebook-dir="/mnt/cee-lgarrow/my-workspace"`
  By using `nohup` the server should keep running even if your connection drops accidentally (or
  intentionally). If you already had started a Jupyter Lab server instance which you disconnected
  from without explicitly terminating, you can skip this step and just reconnect in the next steps.
- Find the port that your server is running on by running `jupyter lab list` and looking for the
  URL that it prints out. It will be something like `http://localhost:8888/lab?token=...`. If you
  see multiple URLs, look for the one that is pointing to the directory you want to work in. Note
  the PORT_NUMBER and TOKEN from the URL. The PORT_NUMBER is the number after `localhost:` and
  before `/lab`, and the TOKEN is the string after `token=` in the URL. These should be unique to
  your Jupyter Lab instance, so make sure you get the right ones if there are multiple instances
  running. They should also be stable over time for the same Jupyter Lab instance, so you can note
  them down and reuse them in the near future when you reconnect to the same running instance.
  They will change if you start a new Jupyter Lab instance, even if pointing to the same working
  directory, so if you start a new instance make sure to get the new PORT_NUMBER and TOKEN.
- Open a new command prompt (or a new tab on the existing prompt) and run:
  `ssh -L 8999:localhost:PORT_NUMBER YOUR_GT_USER_ID@SERVER_ADDRESS.ce.gatech.edu`, replacing
  `PORT_NUMBER` with the port number you found in the previous step, and `YOUR_GT_USER_ID` and
  `SERVER_ADDRESS` as before. This will create a secure tunnel from your local machine's port 8999
  to the selected port on the server.
- Open a web browser and navigate to `http://localhost:8999/lab?token=TOKEN`, replacing `TOKEN`
  with the token you noted earlier. This should open the Jupyter Lab interface.
