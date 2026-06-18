# Stage 1: Build Vite App
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

# Expose standard port
EXPOSE 80

# Nginx run command
CMD ["nginx", "-g", "daemon off;"]
