# Use Node.js LTS version
FROM --platform=linux/amd64 node:20-alpine

# Set working directory
WORKDIR /src

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy project files
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
