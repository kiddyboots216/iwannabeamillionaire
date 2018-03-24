import React, { Component } from 'react';
import logo from './logo.svg';
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import './App.css';
// import Yee2 from './pagedraw/component_2'
import component2 from './pagedraw/component_2'
import Yee3 from './pagedraw/component_3'
import Yee4 from './pagedraw/component_4'

class App extends Component {
  render() {
    return (
      <div className="App">
        <Router>
          <div>
            <Route exact path="/" component={component2} />
            <Route path="/bamboozle" component={Yee3} />
            <Route path="/ayylmao" component={Yee4} />
          </div>
        </Router>
      </div>
    );
  }
}

export default App;
