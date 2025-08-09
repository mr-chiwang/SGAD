#!/bin/bash
npm run build
git add .
git commit -m "自动部署 $(date +%Y%m%d-%H%M)"
git push origin project_page