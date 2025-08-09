#!/bin/bash
npm run build
git add .
git commit -m "update $(date +%Y%m%d-%H%M)"
git push origin project_page