FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 5173
# Vite dev server — host 0.0.0.0 exposes it outside the container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
