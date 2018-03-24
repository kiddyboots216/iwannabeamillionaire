import {Service} from 'react-services-injector';

class ServiceBoy extends Service {
    constructor() {
    }
  
    changeNumber() {
      this.randomNumber = Math.random();
    }
  
    get number() {
      //we can store pure data and format it in getters
      return Math.floor(this.randomNumber * 100);
    }
  }
  
  //"publicName" property is important if you use any kind of minimization on your JS
  ServiceBoy.publicName = 'ServiceBoy';
  
  export default ServiceBoy;