FROM node:23-slim AS build
LABEL authors="elbekbakiev"

WORKDIR /app
COPY ./frontend/ /app/
RUN npm install
RUN npm run build


FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
