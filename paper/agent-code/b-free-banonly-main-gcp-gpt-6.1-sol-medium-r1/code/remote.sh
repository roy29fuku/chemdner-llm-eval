#!/bin/bash
exec ssh -i /home/agent/.ssh/cem_key -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o 'ProxyCommand=nc -X connect -x proxy:8080 %h %p' agent@136.80.201.181 "$@"
