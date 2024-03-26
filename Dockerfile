FROM andrewmackrodt/nodejs

WORKDIR /app

COPY . .

RUN yarn install

CMD ["yarn", "run", "test"]
